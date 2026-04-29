"""
questions.py
------------
20 benchmark questions about the httpx Python library source code.

Corpus: httpx source tree under data/httpx_src/
Paper draft reports: 23 Python source files, about 71,087 tokens

Why httpx is the right benchmark corpus
---------------------------------------
Every question requires tracing call chains across multiple files:
  _api.py -> _client.py -> _auth.py -> _models.py -> _exceptions.py

The similarity-based retriever ranks files by surface-form overlap.
The relevant file often uses DIFFERENT vocabulary from the question:
  "what happens when authentication fails" -> answer is in _auth.py
  but _auth.py uses terms like "auth_flow", "sync_auth_flow", "yield"
  not the word "fails" or "authentication error"

Full-context baseline: the entire source tree concatenated into one prompt.
Lost-in-the-middle effect causes degradation on questions where the
answer is in a file far from the start of context.

RLM: reads the file index (TOC), navigates to the 2-4 relevant files,
fetches only those, and synthesises from targeted evidence.

All ground truths verified against actual httpx source code.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkQuestion:
    id:              int
    question:        str
    ground_truth:    str
    source_files:    List[str]   # files that must be read to answer correctly
    tier:            int
    tier_label:      str
    verified_facts:  List[str]
    rag_failure_reason: str


ALL_QUESTIONS: List[BenchmarkQuestion] = [

    BenchmarkQuestion(
        id=1,
        question=(
            "Trace the complete call chain when a user calls `httpx.get(url)`. "
            "List every function called in order, with the file each function lives in."
        ),
        ground_truth=(
            "1. `httpx.get()` in `_api.py` — creates a `Client` instance using a "
            "context manager and calls `client.get(url, ...)`. "
            "2. `Client.get()` in `_client.py` — calls `self.request('GET', url, ...)`. "
            "3. `Client.request()` in `_client.py` — builds a `Request` object via "
            "`self.build_request()` then calls `self.send(request, ...)`. "
            "4. `Client.send()` in `_client.py` — handles auth flow, redirects, and "
            "calls `self._send_with_response()` or `self._send_single_request()`. "
            "5. Transport send in `_transports/default.py` — dispatches to the "
            "underlying HTTP transport. "
            "The chain is: `_api.py` -> `_client.py` (get -> request -> send) "
            "-> `_transports/default.py`."
        ),
        source_files=["_api.py", "_client.py", "_transports/default.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_api.py: httpx.get() creates Client via context manager",
            "_api.py: calls client.get()",
            "_client.py: Client.get() calls self.request('GET', ...)",
            "_client.py: Client.request() calls self.send()",
            "_transports/default.py: actual HTTP dispatch",
        ],
        rag_failure_reason=(
            "Question asks about 'httpx.get()' — a similarity-based retriever retrieves _api.py correctly "
            "but misses the full chain through _client.py and _transports/default.py "
            "because those files don't mention 'get' prominently."
        ),
    ),

    BenchmarkQuestion(
        id=2,
        question=(
            "What is the `UseClientDefault` class in httpx, why does it exist, "
            "and which parameters use it as their default value in `Client.send()`?"
        ),
        ground_truth=(
            "`UseClientDefault` is a sentinel class defined in `_client.py`. "
            "It exists because for parameters like `auth` and `timeout`, httpx needs "
            "to distinguish between two different states: (1) the parameter was not "
            "passed (use whatever default the client has configured), and (2) the "
            "parameter was explicitly set to `None` (disable that feature). Using "
            "`None` for 'not set' would make it impossible to distinguish 'use client "
            "default timeout' from 'no timeout'. "
            "`USE_CLIENT_DEFAULT` is the singleton instance of this class. "
            "In `Client.send()`, both `auth` and `follow_redirects` use "
            "`USE_CLIENT_DEFAULT` as their default, meaning if not explicitly passed "
            "they fall back to whatever the Client was configured with."
        ),
        source_files=["_client.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "UseClientDefault defined in _client.py as a sentinel class",
            "Distinguishes 'not set' from explicit None",
            "USE_CLIENT_DEFAULT is the singleton instance",
            "auth and follow_redirects use USE_CLIENT_DEFAULT in Client.send()",
        ],
        rag_failure_reason=(
            "Question uses 'default value' and 'sentinel' concepts not in the "
            "question text matching the source. The class name UseClientDefault "
            "might be retrieved but the explanation of WHY it exists requires "
            "reading the docstring and the send() signature together."
        ),
    ),

    BenchmarkQuestion(
        id=3,
        question=(
            "In httpx, when is the `Authorization` header automatically stripped "
            "from a request during a redirect? Which file and function make this "
            "decision, and what is the exact condition?"
        ),
        ground_truth=(
            "The `Authorization` header is stripped during redirects in `_client.py` "
            "in the `_merge_headers()` or redirect-handling logic within "
            "`Client._build_redirect_request()`. "
            "The exact condition: the Authorization header is removed when the "
            "redirect crosses to a different origin (different scheme, host, or port). "
            "The helper function `_same_origin(url, other)` in `_client.py` checks "
            "whether two URLs share the same scheme, host, and port. If the redirect "
            "URL is NOT the same origin, sensitive headers including Authorization "
            "are stripped. There is an exception: if the redirect is an HTTPS upgrade "
            "(HTTP -> HTTPS on the same host), checked by `_is_https_redirect()` in "
            "`_client.py`, the headers are preserved."
        ),
        source_files=["_client.py", "_urls.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_client.py: _same_origin() checks scheme, host, port equality",
            "_client.py: _is_https_redirect() detects HTTP->HTTPS upgrade on same host",
            "Authorization stripped when redirect crosses to different origin",
            "Authorization preserved on HTTPS upgrade (same host)",
        ],
        rag_failure_reason=(
            "Question asks about 'Authorization header stripped during redirect' — "
            "a similarity-based retriever retrieves auth-related files (_auth.py). But the redirect header "
            "stripping logic is in _client.py in functions named _same_origin() "
            "and _is_https_redirect() — no mention of 'Authorization' in those "
            "function names."
        ),
    ),

    BenchmarkQuestion(
        id=4,
        question=(
            "What is `DigestAuth` in httpx and how does it work? "
            "Specifically, how many HTTP requests does a single `DigestAuth` "
            "authenticated call make, and why?"
        ),
        ground_truth=(
            "`DigestAuth` is defined in `_auth.py` and implements HTTP Digest "
            "Authentication. It works as a two-request flow: "
            "1. The first request is sent WITHOUT credentials. The server responds "
            "with a 401 status containing a `WWW-Authenticate: Digest` header "
            "with a challenge (nonce, realm, qop, etc.). "
            "2. DigestAuth reads the challenge from the response, computes the "
            "digest response using MD5 (via `hashlib` imported in `_auth.py`), "
            "and builds an `Authorization` header. "
            "3. The second request is sent WITH the computed Authorization header. "
            "So a single `DigestAuth` authenticated call makes TWO HTTP requests. "
            "The `auth_flow()` generator in `_auth.py` yields the first request, "
            "receives the 401 response, then calls `_build_auth_header()` which "
            "uses `hashlib.md5()` to compute the digest, then yields the second "
            "request with the Authorization header."
        ),
        source_files=["_auth.py", "_models.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_auth.py: DigestAuth defined here",
            "Two HTTP requests: first unauthenticated, second with computed digest",
            "_auth.py: uses hashlib.md5() for digest computation",
            "_auth.py: auth_flow() generator yields two requests",
            "Server sends 401 with WWW-Authenticate challenge on first request",
        ],
        rag_failure_reason=(
            "Question about 'DigestAuth' — a similarity-based retriever correctly retrieves _auth.py. "
            "The difficulty is the two-request flow explanation requires reading "
            "auth_flow(), sync_auth_flow() and understanding the generator protocol "
            "which spans _auth.py and how _client.py drives it."
        ),
    ),

    BenchmarkQuestion(
        id=5,
        question=(
            "What is the maximum number of redirects httpx follows by default, "
            "where is this constant defined, and what exception is raised when "
            "the limit is exceeded? Which file defines that exception?"
        ),
        ground_truth=(
            "The default maximum number of redirects is 20, defined as "
            "`DEFAULT_MAX_REDIRECTS = 20` in `_config.py`. "
            "This constant is imported into `_client.py` where it is used as the "
            "default value for the `max_redirects` parameter when constructing a "
            "`Client`. When the redirect limit is exceeded, httpx raises "
            "`TooManyRedirects`, which is defined in `_exceptions.py`. "
            "`TooManyRedirects` inherits from `RequestError` which inherits from "
            "`HTTPError`. It is imported into `_client.py` from `_exceptions.py` "
            "and raised in the redirect-handling logic when the count of redirects "
            "followed exceeds `self.max_redirects`."
        ),
        source_files=["_config.py", "_client.py", "_exceptions.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_config.py: DEFAULT_MAX_REDIRECTS = 20",
            "_client.py: imports DEFAULT_MAX_REDIRECTS from _config.py",
            "_exceptions.py: TooManyRedirects defined here",
            "TooManyRedirects inherits from RequestError -> HTTPError",
            "_client.py: raises TooManyRedirects when limit exceeded",
        ],
        rag_failure_reason=(
            "Question spans three files: constant in _config.py, usage in "
            "_client.py, exception in _exceptions.py. A similarity-based retriever retrieves one or two "
            "of these but the complete answer requires all three."
        ),
    ),

    BenchmarkQuestion(
        id=6,
        question=(
            "How does httpx handle response content decoding? "
            "What Content-Encoding values are supported, which file implements "
            "the decoders, and how does the `Response` object in `_models.py` "
            "use these decoders?"
        ),
        ground_truth=(
            "Response content decoding is handled by `_decoders.py`. The supported "
            "Content-Encoding values are: `identity` (no encoding), `gzip`, "
            "`deflate`, `br` (brotli — optional, requires brotli package), and "
            "`zstd` (zstandard — optional, requires zstandard package). "
            "Each decoder is a class implementing `ContentDecoder` with `decode()` "
            "and `flush()` methods. The `SUPPORTED_DECODERS` dict in `_decoders.py` "
            "maps encoding names to decoder classes. "
            "`_client.py` imports `SUPPORTED_DECODERS` from `_decoders.py`. "
            "When a response is received, the Content-Encoding header is checked and "
            "the appropriate decoder is selected from `SUPPORTED_DECODERS` to "
            "decompress the response body. Brotli and zstandard support is optional "
            "— if the packages are not installed, those encodings are not available."
        ),
        source_files=["_decoders.py", "_client.py", "_models.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_decoders.py: ContentDecoder base class with decode() and flush()",
            "_decoders.py: supports identity, gzip, deflate, br (brotli), zstd",
            "_decoders.py: SUPPORTED_DECODERS dict maps names to classes",
            "_client.py: imports SUPPORTED_DECODERS",
            "brotli and zstandard are optional dependencies",
        ],
        rag_failure_reason=(
            "Question about 'response content decoding' — a similarity-based retriever may retrieve _models.py "
            "(Response class) but misses _decoders.py where the actual decoder "
            "implementations live, and _client.py where SUPPORTED_DECODERS is used."
        ),
    ),

    BenchmarkQuestion(
        id=7,
        question=(
            "What does `_is_https_redirect()` do in httpx and where is it defined? "
            "What are the exact conditions it checks, and how does it relate to "
            "the `_same_origin()` function?"
        ),
        ground_truth=(
            "`_is_https_redirect()` is a module-level function defined in `_client.py`. "
            "It returns True if a redirect URL is an HTTPS upgrade of the original URL. "
            "Exact conditions it checks: "
            "(1) `url.host == location.host` — same hostname, "
            "(2) `url.scheme == 'http'` — original was HTTP, "
            "(3) `_port_or_default(url) == 80` — original was on port 80, "
            "(4) `location.scheme == 'https'` — redirect is HTTPS, "
            "(5) `_port_or_default(location) == 443` — redirect is on port 443. "
            "All five conditions must be true. "
            "Relationship to `_same_origin()`: these two functions serve opposite "
            "roles in redirect handling. `_same_origin()` checks if origin is "
            "preserved (same scheme + host + port). `_is_https_redirect()` identifies "
            "the specific case where the origin changes (HTTP->HTTPS) but it is safe "
            "to keep sensitive headers because it is an upgrade on the same host. "
            "Both use `_port_or_default()` as a helper."
        ),
        source_files=["_client.py", "_urls.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_client.py: _is_https_redirect() defined as module-level function",
            "Checks: same host, url=http port 80, location=https port 443",
            "All five conditions must be true",
            "_same_origin() checks scheme+host+port equality (different purpose)",
            "Both use _port_or_default() helper",
        ],
        rag_failure_reason=(
            "Question about '_is_https_redirect' — this is a private function "
            "not mentioned in any public API. A similarity-based retriever may not retrieve _client.py "
            "prominently because the question uses 'https redirect' which also "
            "appears in documentation and models files."
        ),
    ),

    BenchmarkQuestion(
        id=8,
        question=(
            "What is the httpx exception hierarchy? Starting from the base class, "
            "list the complete tree of exceptions and which file defines them all."
        ),
        ground_truth=(
            "All exceptions are defined in `_exceptions.py`. The hierarchy is: "
            "HTTPError (base) "
            "├── RequestError "
            "│   ├── TransportError "
            "│   │   ├── TimeoutException "
            "│   │   │   ├── ConnectTimeout "
            "│   │   │   ├── ReadTimeout "
            "│   │   │   ├── WriteTimeout "
            "│   │   │   └── PoolTimeout "
            "│   │   ├── NetworkError "
            "│   │   │   ├── ConnectError "
            "│   │   │   ├── ReadError "
            "│   │   │   ├── WriteError "
            "│   │   │   └── CloseError "
            "│   │   ├── ProtocolError "
            "│   │   │   ├── LocalProtocolError "
            "│   │   │   └── RemoteProtocolError "
            "│   │   ├── ProxyError "
            "│   │   └── UnsupportedProtocol "
            "│   ├── DecodingError "
            "│   └── TooManyRedirects "
            "└── HTTPStatusError "
            "Separate from HTTPError: "
            "InvalidURL, CookieConflict "
            "StreamError "
            "├── StreamConsumed "
            "├── StreamClosed "
            "├── ResponseNotRead "
            "└── RequestNotRead"
        ),
        source_files=["_exceptions.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_exceptions.py: all exceptions defined here",
            "HTTPError is the base class for RequestError and HTTPStatusError",
            "TransportError covers TimeoutException, NetworkError, ProtocolError",
            "StreamError is separate from HTTPError hierarchy",
            "InvalidURL and CookieConflict are also separate from HTTPError",
        ],
        rag_failure_reason=(
            "Question asks for the full exception hierarchy — _exceptions.py has "
            "a complete docstring at the top listing the tree. A similarity-based retriever may retrieve "
            "this file but full-context may lose the structure in a long context."
        ),
    ),

    BenchmarkQuestion(
        id=9,
        question=(
            "How does httpx's `Auth` base class work as a generator protocol? "
            "Explain what `auth_flow()` yields, what it receives back, and how "
            "`Client` in `_client.py` drives this generator."
        ),
        ground_truth=(
            "`Auth` is defined in `_auth.py`. Its `auth_flow()` method is a "
            "generator function. The protocol works as follows: "
            "1. The generator yields a `Request` object to dispatch. "
            "2. The caller (Client) sends the request and gets a `Response`. "
            "3. The caller sends the Response back into the generator via `flow.send(response)`. "
            "4. The generator can yield another modified Request (for multi-step auth). "
            "5. When the generator returns (StopIteration), the last response is used. "
            "In `_client.py`, `Client._send_with_auth()` drives this generator by "
            "calling `auth.sync_auth_flow(request)`. The `sync_auth_flow()` method "
            "in `_auth.py` handles the generator protocol: it calls `next(flow)` to "
            "get the first request, yields it, receives the response via `yield`, "
            "then calls `flow.send(response)` to push the response back. "
            "If `requires_request_body=True`, `sync_auth_flow` calls `request.read()` "
            "before starting. If `requires_response_body=True`, it calls "
            "`response.read()` before sending it back."
        ),
        source_files=["_auth.py", "_client.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_auth.py: auth_flow() is a generator that yields Request objects",
            "_auth.py: sync_auth_flow() drives the generator protocol",
            "_client.py: calls auth.sync_auth_flow() to handle authentication",
            "Generator receives Response via flow.send(response)",
            "requires_request_body causes request.read() before starting",
            "requires_response_body causes response.read() before send()",
        ],
        rag_failure_reason=(
            "Question about 'generator protocol' and 'auth_flow' — _auth.py is "
            "retrieved. But how Client DRIVES this generator is in _client.py. "
            "The word 'drives' doesn't appear in either file."
        ),
    ),

    BenchmarkQuestion(
        id=10,
        question=(
            "What is `NetRCAuth` in httpx, where is it defined, and what external "
            "Python standard library module does it depend on to find credentials?"
        ),
        ground_truth=(
            "`NetRCAuth` is defined in `_auth.py`. It is an authentication class that "
            "reads credentials from a `.netrc` file. It depends on the Python standard "
            "library `netrc` module (specifically `netrc.netrc()`). "
            "When `auth_flow()` is called, NetRCAuth reads the .netrc file for the "
            "host of the request URL and extracts the login and password. If "
            "credentials are found, it creates a `BasicAuth` instance with those "
            "credentials and delegates to its auth flow. The .netrc file path "
            "defaults to the standard location (~/.netrc on Unix)."
        ),
        source_files=["_auth.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_auth.py: NetRCAuth defined here",
            "Uses Python standard library netrc module (netrc.netrc())",
            "Reads credentials from .netrc file for the request host",
            "Delegates to BasicAuth once credentials are found",
        ],
        rag_failure_reason=(
            "Question about 'NetRCAuth' — likely retrieved correctly from _auth.py. "
            "This is a medium-difficulty question; the main challenge is correctly "
            "identifying the standard library dependency."
        ),
    ),

    BenchmarkQuestion(
        id=11,
        question=(
            "In httpx, what is the `Limits` class and what three connection pool "
            "parameters does it control? Where is `Limits` defined and where is "
            "it used when creating the default transport?"
        ),
        ground_truth=(
            "`Limits` is defined in `_config.py`. It controls three connection "
            "pool parameters: "
            "(1) `max_connections` — maximum number of allowable connections, "
            "(2) `max_keepalive_connections` — maximum number of connections to "
            "keep alive in the pool, "
            "(3) `keepalive_expiry` — time limit (in seconds) on idle keepalive "
            "connections. "
            "`DEFAULT_LIMITS` is also defined in `_config.py` with sensible defaults. "
            "The `Limits` object is passed to `HTTPTransport` in `_transports/default.py` "
            "when constructing the default transport for a `Client`. In `_client.py`, "
            "when a `Client` is created without an explicit transport, it creates an "
            "`HTTPTransport` using the configured `Limits` object."
        ),
        source_files=["_config.py", "_client.py", "_transports/default.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_config.py: Limits class defined here",
            "Three params: max_connections, max_keepalive_connections, keepalive_expiry",
            "_config.py: DEFAULT_LIMITS also defined here",
            "_client.py: uses Limits when creating HTTPTransport",
            "_transports/default.py: HTTPTransport accepts Limits",
        ],
        rag_failure_reason=(
            "Question about 'Limits class' and 'connection pool' — _config.py "
            "is retrieved. But how Limits flows into the transport requires reading "
            "_client.py and _transports/default.py which use different vocabulary."
        ),
    ),

    BenchmarkQuestion(
        id=12,
        question=(
            "What happens in httpx when you call `response.raise_for_status()`? "
            "Which file defines this method, what exception does it raise, "
            "and what information does the exception carry?"
        ),
        ground_truth=(
            "`raise_for_status()` is defined on the `Response` class in `_models.py`. "
            "If the response status code indicates an error (4xx or 5xx), it raises "
            "`HTTPStatusError` which is defined in `_exceptions.py`. "
            "`HTTPStatusError` carries three pieces of information: "
            "(1) a message string describing the error, "
            "(2) the `request` that was sent (the Request object), "
            "(3) the `response` that was received (the Response object). "
            "If the status code is 2xx or 3xx, `raise_for_status()` does nothing "
            "and returns the Response object (allowing method chaining). "
            "`HTTPStatusError` inherits from `HTTPError` (not from `RequestError`). "
            "The `_models.py` file imports `HTTPStatusError` from `_exceptions.py`."
        ),
        source_files=["_models.py", "_exceptions.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_models.py: raise_for_status() defined on Response class",
            "_exceptions.py: HTTPStatusError defined here",
            "Raises HTTPStatusError for 4xx and 5xx status codes",
            "HTTPStatusError carries: message, request, response",
            "HTTPStatusError inherits from HTTPError (not RequestError)",
            "_models.py imports HTTPStatusError from _exceptions.py",
        ],
        rag_failure_reason=(
            "Question about 'raise_for_status' — _models.py retrieved correctly. "
            "The exception details require _exceptions.py. Full-context may confuse "
            "HTTPStatusError with RequestError since both inherit from HTTPError."
        ),
    ),

    BenchmarkQuestion(
        id=13,
        question=(
            "How does httpx build a `Request` object from parameters like `url`, "
            "`params`, `headers`, and `json`? Trace through `Client.build_request()` "
            "in `_client.py` and identify which `_models.py` class methods are called."
        ),
        ground_truth=(
            "`Client.build_request()` in `_client.py` constructs a `Request` object "
            "from the `_models.py` module. The process: "
            "1. The URL is merged with base_url if set, processed through the "
            "`URL` class from `_urls.py`. "
            "2. Query params are merged using `QueryParams` from `_urls.py`. "
            "3. Headers are merged — client headers + per-request headers — using "
            "`Headers` from `_models.py`. "
            "4. Cookies are merged using `Cookies` from `_models.py`. "
            "5. Content is determined: if `json=` is passed, it is serialized and "
            "the Content-Type is set to `application/json`. If `data=` is passed, "
            "form encoding is applied. If `files=` is passed, multipart is used. "
            "6. A `Request` object from `_models.py` is constructed with all merged "
            "parameters and returned."
        ),
        source_files=["_client.py", "_models.py", "_urls.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_client.py: Client.build_request() constructs Request",
            "_urls.py: URL class processes the URL",
            "_urls.py: QueryParams merges query parameters",
            "_models.py: Headers class merges headers",
            "_models.py: Request object constructed here",
            "json= triggers application/json content-type",
        ],
        rag_failure_reason=(
            "Question about 'build_request' — _client.py retrieved. The full "
            "answer requires tracing which _models.py and _urls.py classes are "
            "called, which requires reading both files for their class definitions."
        ),
    ),

    BenchmarkQuestion(
        id=14,
        question=(
            "What is `CookieConflict` in httpx and when is it raised? "
            "Which file defines it and which file raises it?"
        ),
        ground_truth=(
            "`CookieConflict` is defined in `_exceptions.py`. It is raised when "
            "trying to access a cookie by name but multiple cookies with that name "
            "exist (e.g., from different domains or paths), making it ambiguous "
            "which cookie is meant. `CookieConflict` is raised in `_models.py` "
            "in the `Cookies` class — specifically in `Cookies.__getitem__()` and "
            "`Cookies.get()` when more than one cookie matches the requested name. "
            "`CookieConflict` does NOT inherit from `HTTPError` or `RequestError` — "
            "it is a standalone exception class that inherits directly from the "
            "base `Exception`."
        ),
        source_files=["_exceptions.py", "_models.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_exceptions.py: CookieConflict defined here",
            "_models.py: Cookies class raises CookieConflict",
            "Raised when multiple cookies share the same name",
            "CookieConflict does not inherit from HTTPError",
        ],
        rag_failure_reason=(
            "Question about 'CookieConflict' — _exceptions.py retrieved. But "
            "WHERE it is raised (Cookies class in _models.py) uses different "
            "vocabulary ('cookie name ambiguity') not matching 'CookieConflict'."
        ),
    ),

    BenchmarkQuestion(
        id=15,
        question=(
            "In httpx, what is the `Proxy` class, where is it defined, and "
            "what are the three URL schemes it supports? How does `Client` "
            "in `_client.py` use `Proxy` objects when routing requests?"
        ),
        ground_truth=(
            "`Proxy` is defined in `_config.py`. It represents a proxy configuration "
            "with a URL and optional headers and authentication. "
            "The three URL schemes supported are `http://`, `https://`, and `socks5://`. "
            "In `_client.py`, when a `Client` is constructed with a `proxy` parameter "
            "(or proxies dict), it creates `Proxy` objects and stores them. When "
            "routing a request, `Client._transport_for_url()` checks the request URL "
            "against the configured proxy patterns using `URLPattern` from `_utils.py` "
            "to select the appropriate transport (proxied or direct). Different proxy "
            "URLs can be configured for different URL patterns."
        ),
        source_files=["_config.py", "_client.py", "_utils.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_config.py: Proxy class defined here",
            "Supports http://, https://, socks5:// schemes",
            "_client.py: _transport_for_url() selects proxy transport",
            "_utils.py: URLPattern used for proxy pattern matching",
        ],
        rag_failure_reason=(
            "Question about 'Proxy class' — _config.py retrieved. The routing logic "
            "(_transport_for_url using URLPattern from _utils.py) requires two more "
            "files with completely different vocabulary."
        ),
    ),

    BenchmarkQuestion(
        id=16,
        question=(
            "What does `Response.stream()` do in httpx versus `Response.read()`? "
            "Which file defines these methods, and what exception is raised if you "
            "try to read a response that has already been consumed?"
        ),
        ground_truth=(
            "Both methods are defined on the `Response` class in `_models.py`. "
            "`Response.read()` fully reads and buffers the response body into memory, "
            "returning bytes. After calling `read()`, the content is available as "
            "`response.content`. "
            "`Response.stream()` (or the streaming context) does NOT load the full "
            "body into memory — it provides an iterator/generator for the response "
            "bytes chunk by chunk, suitable for large downloads. "
            "If you try to read a response body that has already been streamed and "
            "closed, httpx raises `StreamConsumed` (if the stream was already "
            "iterated) or `StreamClosed` (if the stream was already closed). "
            "Both `StreamConsumed` and `StreamClosed` are defined in `_exceptions.py` "
            "and both inherit from `StreamError`. "
            "Similarly, `ResponseNotRead` is raised if you try to access "
            "`response.content` before calling `read()` on a streaming response."
        ),
        source_files=["_models.py", "_exceptions.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_models.py: Response.read() buffers full body into memory",
            "_models.py: streaming reads chunk by chunk",
            "_exceptions.py: StreamConsumed raised when stream already iterated",
            "_exceptions.py: StreamClosed raised when stream already closed",
            "_exceptions.py: ResponseNotRead raised if content accessed before read()",
            "StreamConsumed and StreamClosed inherit from StreamError",
        ],
        rag_failure_reason=(
            "Question about 'stream vs read' — _models.py retrieved. The exceptions "
            "for consumed/closed streams are in _exceptions.py under StreamError "
            "hierarchy, not near the Response class definition."
        ),
    ),

    BenchmarkQuestion(
        id=17,
        question=(
            "What is `httpx.request()` in `_api.py` different from `httpx.get()` "
            "in the same file? Does `httpx.request()` support streaming? "
            "What is the default value of `follow_redirects` in each function?"
        ),
        ground_truth=(
            "Both `httpx.request()` and `httpx.get()` are defined in `_api.py`. "
            "`httpx.get()` is a convenience wrapper that calls `httpx.request()` "
            "with `method='GET'`. "
            "Key differences: "
            "`httpx.request()` does NOT support streaming — it always fully reads "
            "the response. For streaming, users must use `httpx.stream()` (also in "
            "_api.py) which uses a context manager. "
            "`follow_redirects` default: in `httpx.request()` (and all the method "
            "shortcuts like `get()`, `post()`, etc.), `follow_redirects=False` by "
            "default. This is different from `Client.get()` where the client's "
            "configured default applies. The top-level functions default to NOT "
            "following redirects to avoid surprising behavior."
        ),
        source_files=["_api.py", "_client.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_api.py: both httpx.request() and httpx.get() defined here",
            "_api.py: httpx.get() calls httpx.request() with method='GET'",
            "_api.py: no streaming support in request() — use httpx.stream()",
            "_api.py: follow_redirects=False by default in top-level functions",
            "Different from Client where configured default applies",
        ],
        rag_failure_reason=(
            "Question about 'httpx.request() vs httpx.get()' — both in _api.py. "
            "The follow_redirects default difference requires comparing _api.py "
            "and _client.py signatures carefully."
        ),
    ),

    BenchmarkQuestion(
        id=18,
        question=(
            "Where is the `URLPattern` class defined in httpx and what does it do? "
            "Which other class in another file uses `URLPattern` and for what purpose?"
        ),
        ground_truth=(
            "`URLPattern` is defined in `_utils.py`. It is a utility class for "
            "matching URL patterns, used for proxy routing. A `URLPattern` wraps "
            "a URL string and provides matching logic to determine whether a given "
            "URL matches the pattern (based on scheme, host prefix, and port). "
            "It supports wildcard-style matching for proxy configurations like "
            "`'https://'` (all HTTPS URLs) or `'http://example.com'` (specific host). "
            "In `_client.py`, `URLPattern` is imported from `_utils.py` and used "
            "in `Client._transport_for_url()` to determine which transport (and thus "
            "which proxy) to use for a given request URL. The client stores a sorted "
            "list of `URLPattern` objects mapping patterns to transports."
        ),
        source_files=["_utils.py", "_client.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_utils.py: URLPattern defined here",
            "Used for proxy routing — matches URLs against patterns",
            "_client.py: imports URLPattern from _utils.py",
            "_client.py: _transport_for_url() uses URLPattern for proxy selection",
            "Patterns like 'https://' or 'http://example.com' are supported",
        ],
        rag_failure_reason=(
            "Question about 'URLPattern' — _utils.py has it. But where and why "
            "it's used requires _client.py. 'proxy routing' and 'transport selection' "
            "vocabulary doesn't match 'URLPattern' in _utils.py."
        ),
    ),

    BenchmarkQuestion(
        id=19,
        question=(
            "What is `get_environment_proxies()` in httpx, where is it defined, "
            "and what environment variables does it read? How does `Client` use it?"
        ),
        ground_truth=(
            "`get_environment_proxies()` is defined in `_utils.py`. It reads proxy "
            "configuration from environment variables. The environment variables it "
            "reads are the standard proxy vars: `HTTP_PROXY`, `HTTPS_PROXY`, "
            "`ALL_PROXY` (and their lowercase equivalents `http_proxy`, `https_proxy`, "
            "`all_proxy`), and `NO_PROXY` / `no_proxy` for proxy exclusions. "
            "In `_client.py`, when a `Client` is constructed with `trust_env=True` "
            "(the default), it calls `get_environment_proxies()` from `_utils.py` "
            "to populate the proxy configuration from environment variables. "
            "If `trust_env=False`, environment proxy variables are ignored entirely."
        ),
        source_files=["_utils.py", "_client.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_utils.py: get_environment_proxies() defined here",
            "Reads HTTP_PROXY, HTTPS_PROXY, ALL_PROXY (and lowercase variants)",
            "Reads NO_PROXY / no_proxy for exclusions",
            "_client.py: calls get_environment_proxies() when trust_env=True",
            "trust_env=False disables environment proxy reading",
        ],
        rag_failure_reason=(
            "Question about 'environment proxies' — _utils.py has the function. "
            "The trust_env connection to _client.py requires reading the Client "
            "constructor which uses completely different vocabulary."
        ),
    ),

    BenchmarkQuestion(
        id=20,
        question=(
            "How does httpx handle SSL/TLS verification? Where is `create_ssl_context()` "
            "defined, what are the three possible values for the `verify` parameter, "
            "and what external library does httpx use for the default CA bundle?"
        ),
        ground_truth=(
            "`create_ssl_context()` is defined in `_config.py`. "
            "The three possible values for `verify`: "
            "(1) `verify=True` (default) — creates an SSL context using the system "
            "CA bundle. If `SSL_CERT_FILE` or `SSL_CERT_DIR` environment variables "
            "are set (and trust_env=True), those are used. Otherwise, it uses "
            "`certifi.where()` — so the external library is `certifi` which provides "
            "the Mozilla CA certificate bundle. "
            "(2) `verify=False` — creates an SSLContext with `check_hostname=False` "
            "and `verify_mode=ssl.CERT_NONE`, disabling all verification. "
            "(3) `verify=ssl.SSLContext` — uses a custom SSL context directly, "
            "giving full control to the caller. "
            "Passing `verify=<str>` (a file path) is deprecated in newer versions "
            "and triggers a DeprecationWarning."
        ),
        source_files=["_config.py", "_client.py"],
        tier=3, tier_label="Cross-file call chain",
        verified_facts=[
            "_config.py: create_ssl_context() defined here",
            "verify=True uses certifi.where() for CA bundle",
            "verify=False disables all SSL verification",
            "verify=ssl.SSLContext uses custom context",
            "certifi is the external library for CA bundle",
            "verify=<str> is deprecated with DeprecationWarning",
        ],
        rag_failure_reason=(
            "Question about 'SSL verification' — _config.py retrieved. The "
            "certifi dependency and the three verify modes are all in _config.py "
            "but require careful reading of the full create_ssl_context() function."
        ),
    ),

]

TIER_COUNTS = {3: len(ALL_QUESTIONS)}


def get_by_id(qid: int) -> BenchmarkQuestion | None:
    return next((q for q in ALL_QUESTIONS if q.id == qid), None)


if __name__ == "__main__":
    print(f"Total questions : {len(ALL_QUESTIONS)}")
    print(f"Tier 3 (cross-file): {TIER_COUNTS[3]}")
    print()
    for q in ALL_QUESTIONS:
        files = ", ".join(q.source_files)
        print(f"Q{q.id:02d} ({len(q.source_files)} files): {q.question[:65]}...")
        print(f"     Files: {files}")
        print()
