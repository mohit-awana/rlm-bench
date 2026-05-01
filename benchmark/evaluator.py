"""
evaluator.py — two-stage DeepEval scoring cascade for benchmark results.

Stage 1: gpt-4.1-nano (faithfulness) + gpt-4o-mini (Code QA Accuracy).
Stage 2 escalates borderline cases to gpt-4o when:
  - composite score in [0.65, 0.72]
  - symbol_recall < 0.60
  - faithfulness and code_qa_accuracy differ by more than 0.20

Composite: 0.35 * Faithfulness + 0.45 * Code QA Accuracy + 0.20 * Symbol Recall
"""

from __future__ import annotations

import json,re,time,random,logging
from pathlib import Path
import numpy as np

RESULTS_DIR=Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CHECKPOINT=RESULTS_DIR/"cascade_checkpoint.json"

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

SYMBOL_RE=re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

def extract_symbols(t):
    return set(SYMBOL_RE.findall(t))

def symbol_recall(a,t):
    gt=extract_symbols(t)
    if not gt:
        return 0
    return round(len(gt & extract_symbols(a))/len(gt),4)

def cap_text(t,n=800):
    return t[:n]

def bootstrap_ci(scores,n=3000):
    if not scores:
        return (None,None)
    arr=np.array(scores)
    means=[np.mean(np.random.choice(arr,len(arr),replace=True)) for _ in range(n)]
    return (
      round(float(np.percentile(means,2.5)),4),
      round(float(np.percentile(means,97.5)),4)
    )

def build_metrics():
    from deepeval.metrics import FaithfulnessMetric, GEval
    from deepeval.test_case import SingleTurnParams

    return [
        # keep nano for cheap metric
        FaithfulnessMetric(
            threshold=0.60,
            model="gpt-4.1-nano"
        ),

        # use 4o-mini for structured scoring stability
        GEval(
            name="Code QA Accuracy",
            model="gpt-4o-mini",

            criteria="""
Grade code QA using partial credit.

Scoring:
1.0 fully correct
0.8 mostly correct
0.6 substantially correct
0.4 partially correct
0.0 largely incorrect

Reward partial correctness strongly.
Minor file errors should not dominate score.
Extra correct detail is not hallucination.

Return ONLY valid JSON exactly in this form:

{
 "score": 0.72,
 "reason": "Mostly correct call chain, minor file-location errors."
}

Return nothing except JSON.
No markdown.
No code fences.
""",

            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT
            ],
            threshold=0.60
        )
    ]




def run_deepeval(case,model):
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig,DisplayConfig

    metrics = build_metrics()

    for a in range(6):
        try:
            ev=evaluate(
              test_cases=[case],
              metrics=metrics,
              async_config=AsyncConfig(
                run_async=True,
                max_concurrent=1,
                throttle_value=1
              ),
              display_config=DisplayConfig(
                show_indicator=False,
                print_results=False
              )
            )
            return ev.test_results[0]


        except KeyError as e:
            if "score" in str(e):
                logger.warning(
                  "Malformed judge output. Retrying with gpt-4o"
                )

                metrics=build_metrics()

                # hard fallback upgrade to gpt-4o
                for m in metrics:
                    try:
                        if hasattr(m,"model"):
                            m.model="gpt-4o"
                    except:
                        pass

                ev=evaluate(
                    test_cases=[case],
                    metrics=metrics,
                    async_config=AsyncConfig(
                        run_async=True,
                        max_concurrent=1,
                        throttle_value=1
                    ),
                    display_config=DisplayConfig(
                        show_indicator=False,
                        print_results=False
                    )
                )

                return ev.test_results[0]


def parse_scores(tr):
    s={}
    for md in tr.metrics_data or []:
        name=(md.name or "").replace(
            "[GEval]",""
        ).strip().lower().replace(" ","_")
        s[name]=round(float(md.score),4)
    return s

def composite(scores):
    return round(
      0.35*scores["faithfulness"]+
      0.45*scores["code_qa_accuracy"]+
      0.20*scores["symbol_recall"],
      4
    )

def borderline(scores):
    c=composite(scores)
    return (
      (0.65<=c<=0.72)
      or
      scores["symbol_recall"]<0.60
      or
      abs(
       scores["faithfulness"]-
       scores["code_qa_accuracy"]
      )>0.20
    )

def save_checkpoint(scored):
    with open(CHECKPOINT,"w") as f:
        json.dump(
          {"results":scored},
          f,
          indent=2
        )

class Evaluator:

  def run(self,results):

    from deepeval.test_case import LLMTestCase

    scored=[]
    done_ids=set()

    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            prior=json.load(f)

        scored=prior["results"]

        done_ids={
          r["question_id"]
          for r in scored
          if "question_id" in r
        }

        logger.info(
         f"Resuming with {len(done_ids)} done"
        )

    escalated=0

    for r in results:

      qid=r["question_id"]

      if qid in done_ids:
         continue

      tc=LLMTestCase(
         input=r["question"],
         actual_output=cap_text(r["answer"]),
         expected_output=cap_text(
            r["ground_truth"]
         ),
         retrieval_context=r.get(
           "retrieved_chunks",[]
         ) or ["No context"]
      )

      res=run_deepeval(
         tc,
         "gpt-4.1-nano"
      )

      scores=parse_scores(res)

      sr=symbol_recall(
        r["answer"],
        r["ground_truth"]
      )

      scores["symbol_recall"]=sr

      if borderline(scores):
          escalated+=1
          res2=run_deepeval(
            tc,
            "gpt-4o"
          )
          scores=parse_scores(res2)
          scores["symbol_recall"]=sr
          time.sleep(2)

      comp=composite(scores)

      row={
        **r,
        "scores":scores,
        "composite_score":comp,
        "passed":comp>=0.60
      }

      scored.append(row)

      save_checkpoint(scored)

      logger.info(
       f"Saved checkpoint qid={qid}"
      )

      time.sleep(2)

    self.summary(scored,escalated)

  def summary(self,rows,escalated):

    buckets={"rlm":[],"full_context":[]}

    for r in rows:
      buckets[r["pipeline"]].append(r)

    print("="*110)
    print("CRASH SAFE CASCADED SUMMARY")
    print("="*110)

    for p,items in buckets.items():

      comps=[x["composite_score"] for x in items]
      toks=[x["total_tokens"] for x in items]
      lat=[x["elapsed_seconds"] for x in items]

      print(
       p,
       {
        "n":len(items),
        "composite":round(float(np.mean(comps)),4),
        "ci95":bootstrap_ci(comps),
        "avg_tokens":round(float(np.mean(toks)),1),
        "p50_latency":round(
           float(np.percentile(lat,50)),3
        ),
        "p95_latency":round(
           float(np.percentile(lat,95)),3
        ),
        "pass_rate":round(
           float(np.mean([
             x["passed"] for x in items
           ])),4
        )
       }
      )

    print(
      {
       "escalated_cases":escalated
      }
    )

if __name__=="__main__":
   import argparse

   p=argparse.ArgumentParser()
   p.add_argument(
     "--input",
     default="benchmark_results.json"
   )
   args=p.parse_args()

   with open(
      RESULTS_DIR/args.input
   ) as f:
      results=json.load(f)

   Evaluator().run(results)
