"""
LangChain FastAPI Application with Vercel AI Gateway
Vercel Zero Config FastAPI Support with Vercel AI Gateway Integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import httpx
import json
import re
import ast
import io
import math
import cmath
import asyncio
from contextlib import redirect_stdout
import time
import statistics
import fractions
import decimal

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LangChain API with Vercel AI Gateway",
    description="FastAPI application using Vercel AI Gateway for AI inference",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    system_prompt: Optional[str] = "You are a helpful assistant."
    temperature: Optional[float] = 0.7


class QueryResponse(BaseModel):
    query: str
    response: str
    model: str
    traces: Optional[List[Dict[str, Any]]] = None


# Vercel AI Gateway Configuration (env-based)
VERCEL_AI_GATEWAY_URL = os.getenv(
    "VERCEL_AI_GATEWAY_URL",
    "https://ai-gateway.vercel.sh/v1/chat/completions"
)

# Model (cheap GPT‑5 variant via Gateway). Override with env AI_MODEL if desired.
MODEL_NAME = os.getenv("AI_MODEL", "openai/gpt-5-mini")


# (no helper functions; logic is inlined in endpoints)


# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "LangChain FastAPI is running",
        "endpoints": {
            "query": "/api/query",
            "docs": "/docs"
        }
    }


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the AI model using Vercel AI Gateway with math detection and safe Python execution.
    Returns detailed traces including prompt/response previews and token usage when available.
    """
    try:
        traces: List[Dict[str, Any]] = []
        t0 = time.perf_counter()

        # Build messages
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.query}
        ]

        # Detect math-like queries
        detect_t = time.perf_counter()
        t = (request.query or "").lower()
        keywords = (
            "integral", "derivative", "differentiate", "limit", "probability", "matrix",
            "vector", "eigenvalue", "eigenvector", "sum", "product", "series", "log",
            "ln", "sin", "cos", "tan", "sqrt", "expectation", "variance", "median",
            "mean", "std", "standard deviation", "optimize", "minimize", "maximize"
        )
        hits = [k for k in keywords if k in t]
        op_hit = bool(re.search(r"[0-9][^a-zA-Z]*[+\-*/^]", t))
        is_math = bool(hits) or op_hit
        traces.append({
            "phase": "detect", "action": "classify",
            "duration_ms": int((time.perf_counter()-detect_t)*1000),
            "meta": {"is_math": is_math, "hits": hits + (["operator"] if op_hit else [])}
        })

        gateway_key = os.getenv("AI_GATEWAY_API_KEY")
        if not gateway_key:
            raise HTTPException(status_code=500, detail="AI Gateway key not configured. Link your project to Vercel AI Gateway to auto-inject AI_GATEWAY_API_KEY.")

        if is_math:
            # Ask model to produce safe Python code
            system = (
                "You write minimal, safe Python to solve math. Use only the math module; "
                "no other imports, no I/O, no files, no network. Print the final answer."
            )
            user = (
                f"Question: {request.query}\n\n"
                "Write Python that computes the answer and prints it. Return only a Python code block."
            )
            traces.append({
                "phase": "code_gen", "action": "prompt", "duration_ms": 0,
                "meta": {"system_preview": system[:200], "user_preview": user[:200]}
            })
            cg_t = time.perf_counter()
            async with httpx.AsyncClient() as client:
                code_resp = await client.post(
                    VERCEL_AI_GATEWAY_URL,
                    json={
                        "model": MODEL_NAME,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        "temperature": max(0.0, min(request.temperature or 0.3, 0.7)),
                        "max_tokens": 800,
                    },
                    headers={"Authorization": f"Bearer {gateway_key}", "Content-Type": "application/json"},
                    timeout=30.0,
                )
            traces.append({
                "phase": "code_gen", "action": "gateway_request",
                "duration_ms": int((time.perf_counter()-cg_t)*1000),
                "meta": {"model": MODEL_NAME, "status_code": code_resp.status_code}
            })
            if code_resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"AI Gateway code generation failed: {code_resp.text}")
            code_json = code_resp.json()
            code_text = code_json["choices"][0]["message"]["content"]
            traces.append({
                "phase": "code_gen", "action": "response_preview", "duration_ms": 0,
                "meta": {"preview": (code_text[:240] + "…") if len(code_text) > 240 else code_text}
            })
            if isinstance(code_json, dict) and isinstance(code_json.get("usage"), dict):
                u = code_json["usage"]
                traces.append({
                    "phase": "code_gen", "action": "usage", "duration_ms": 0,
                    "meta": {"tokens_prompt": u.get("prompt_tokens"), "tokens_completion": u.get("completion_tokens"), "tokens_total": u.get("total_tokens")}
                })

            # Extract code from possible fenced block
            m = re.search(r"```(?:python)?\s*([\s\S]*?)```", code_text)
            code = (m.group(1) if m else code_text).strip()
            traces.append({
                "phase": "code_gen", "action": "received", "duration_ms": 0,
                "meta": {"length": len(code)}
            })

            # Safety checks and sanitization
            sanitize_t = time.perf_counter()
            lowered = code.lower()
            banned = ["open(", "eval(", "exec(", "os.", "sys.", "subprocess", "socket", "requests", "httpx", "urllib"]
            if any(b in lowered for b in banned):
                raise HTTPException(status_code=400, detail="Unsafe code detected in generated solution.")
            try:
                tree = ast.parse(code)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse generated code: {e}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in ("math", "cmath", "statistics", "fractions", "decimal"):
                            raise HTTPException(status_code=400, detail="Only 'math', 'cmath', 'statistics', 'fractions', and 'decimal' imports are allowed.")
                if isinstance(node, ast.ImportFrom):
                    if (node.module or "") not in ("math", "cmath", "statistics", "fractions", "decimal"):
                        raise HTTPException(status_code=400, detail="Only 'from math|cmath|statistics|fractions|decimal import ...' is allowed.")
                if isinstance(node, ast.Attribute) and isinstance(node.attr, str) and node.attr.startswith("__"):
                    raise HTTPException(status_code=400, detail="Dunder attributes are not allowed.")
            injected: Dict[str, Any] = {}
            new_lines: List[str] = []
            for line in code.splitlines():
                m_from_math = re.match(r"^\s*from\s+math\s+import\s+(.+)$", line)
                m_from_cmath = re.match(r"^\s*from\s+cmath\s+import\s+(.+)$", line)
                m_from_statistics = re.match(r"^\s*from\s+statistics\s+import\s+(.+)$", line)
                m_from_fractions = re.match(r"^\s*from\s+fractions\s+import\s+(.+)$", line)
                m_from_decimal = re.match(r"^\s*from\s+decimal\s+import\s+(.+)$", line)
                m_imp_math = re.match(r"^\s*import\s+math\s*$", line)
                m_imp_cmath = re.match(r"^\s*import\s+cmath\s*$", line)
                m_imp_statistics = re.match(r"^\s*import\s+statistics\s*$", line)
                m_imp_fractions = re.match(r"^\s*import\s+fractions\s*$", line)
                m_imp_decimal = re.match(r"^\s*import\s+decimal\s*$", line)
                from_match = m_from_math or m_from_cmath or m_from_statistics or m_from_fractions or m_from_decimal
                if from_match:
                    names = [p.strip() for p in from_match.group(1).split(',')]
                    module_obj = (
                        math if m_from_math else
                        cmath if m_from_cmath else
                        statistics if m_from_statistics else
                        fractions if m_from_fractions else
                        decimal
                    )
                    for name in names:
                        base = name.split(' as ')[0].strip()
                        if hasattr(module_obj, base):
                            injected[base] = getattr(module_obj, base)
                    continue
                if m_imp_math or m_imp_cmath or m_imp_statistics or m_imp_fractions or m_imp_decimal:
                    # Drop explicit imports; modules provided in globals
                    continue
                new_lines.append(line)
            code = "\n".join(new_lines)

            # Capture last expression and last assignment targets for fallback evaluation
            last_expr_src: Optional[str] = None
            last_assign_vars: List[str] = []
            try:
                sanitized_tree = ast.parse(code)
                if sanitized_tree.body and isinstance(sanitized_tree.body[-1], ast.Expr):
                    last_expr = sanitized_tree.body[-1].value
                    if hasattr(ast, 'unparse'):
                        last_expr_src = ast.unparse(last_expr)
                # find last assignment statement targets
                for node in reversed(sanitized_tree.body):
                    if isinstance(node, ast.Assign):
                        for tgt in node.targets:
                            if isinstance(tgt, ast.Name):
                                last_assign_vars.append(tgt.id)
                        if last_assign_vars:
                            break
            except Exception:
                last_expr_src = None

            preview = (code[:240] + "…") if len(code) > 240 else code
            traces.append({
                "phase": "sanitize", "action": "strip_imports",
                "duration_ms": int((time.perf_counter()-sanitize_t)*1000),
                "meta": {"preview": preview}
            })

            # Execute in restricted sandbox
            def _runner() -> str:
                safe_builtins = {
                    "abs": abs, "min": min, "max": max, "round": round, "range": range, "len": len, "sum": sum, "pow": pow,
                    "print": print, "int": int, "float": float, "complex": complex, "bool": bool, "str": str,
                    "list": list, "tuple": tuple, "set": set, "dict": dict, "sorted": sorted, "enumerate": enumerate,
                    "zip": zip, "map": map, "filter": filter, "any": any, "all": all, "divmod": divmod,
                    "bin": bin, "hex": hex, "oct": oct
                }
                globals_dict = {"__builtins__": safe_builtins, "math": math, "cmath": cmath, "statistics": statistics, "fractions": fractions, "decimal": decimal}
                globals_dict.update(injected)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    exec(compile(code, "<math_solver>", "exec"), globals_dict, {})
                return buf.getvalue().strip()

            exec_t = time.perf_counter()
            stdout = await asyncio.wait_for(asyncio.to_thread(_runner), timeout=10.0)
            if not stdout:
                # Fallback: evaluate last expression or common variable names
                fallback_t = time.perf_counter()
                val: Optional[str] = None
                candidates: List[str] = list(dict.fromkeys((last_assign_vars or []) + [
                    "result", "ans", "answer", "x", "root", "roots", "solution", "sol", "y"
                ]))
                traces.append({
                    "phase": "exec", "action": "fallback_prepare", "duration_ms": 0,
                    "meta": {"has_last_expr": bool(last_expr_src), "last_assign_vars": last_assign_vars, "candidates": candidates}
                })
                if last_expr_src:
                    try:
                        def _eval_last() -> str:
                            safe_builtins = {
                                "abs": abs, "min": min, "max": max, "round": round, "range": range, "len": len, "sum": sum, "pow": pow,
                                "print": print, "int": int, "float": float, "complex": complex, "bool": bool, "str": str,
                                "list": list, "tuple": tuple, "set": set, "dict": dict, "sorted": sorted, "enumerate": enumerate,
                                "zip": zip, "map": map, "filter": filter, "any": any, "all": all, "divmod": divmod,
                                "bin": bin, "hex": hex, "oct": oct
                            }
                            globals_dict = {"__builtins__": safe_builtins, "math": math, "cmath": cmath, "statistics": statistics, "fractions": fractions, "decimal": decimal}
                            globals_dict.update(injected)
                            return str(eval(compile(ast.parse(last_expr_src, mode='eval'), "<math_expr>", "eval"), globals_dict, {}))
                        val = await asyncio.wait_for(asyncio.to_thread(_eval_last), timeout=1.0)
                    except Exception:
                        val = None
                if val is None:
                    for candidate in candidates:
                        try:
                            def _get_var() -> str:
                                safe_builtins = {
                                    "abs": abs, "min": min, "max": max, "round": round, "range": range, "len": len, "sum": sum, "pow": pow,
                                    "print": print, "int": int, "float": float, "complex": complex, "bool": bool, "str": str,
                                    "list": list, "tuple": tuple, "set": set, "dict": dict, "sorted": sorted, "enumerate": enumerate,
                                    "zip": zip, "map": map, "filter": filter, "any": any, "all": all, "divmod": divmod,
                                    "bin": bin, "hex": hex, "oct": oct
                                }
                                globals_dict = {"__builtins__": safe_builtins, "math": math, "cmath": cmath, "statistics": statistics, "fractions": fractions, "decimal": decimal}
                                globals_dict.update(injected)
                                return str(globals_dict.get(candidate))
                            got = await asyncio.wait_for(asyncio.to_thread(_get_var), timeout=0.5)
                            if got not in (None, "None"):
                                val = got
                                break
                        except Exception:
                            pass
                if val is None:
                    traces.append({
                        "phase": "exec", "action": "fallback_failed", "duration_ms": int((time.perf_counter()-fallback_t)*1000),
                        "meta": {"reason": "no_stdout_and_no_value", "has_last_expr": bool(last_expr_src), "last_assign_vars": last_assign_vars}
                    })
                    raise HTTPException(status_code=500, detail="Generated code produced no output.")
                traces.append({
                    "phase": "exec", "action": "fallback_eval",
                    "duration_ms": int((time.perf_counter()-fallback_t)*1000),
                    "meta": {"expr": (last_expr_src or ""), "value": val}
                })
                stdout = val

            traces.append({
                "phase": "exec", "action": "run_python",
                "duration_ms": int((time.perf_counter()-exec_t)*1000),
                "meta": {"stdout": stdout, "stdout_length": len(stdout)}
            })
            response_text = f"Result: {stdout}"

        else:
            # Standard LLM call path
            traces.append({
                "phase": "llm", "action": "prompt", "duration_ms": 0,
                "meta": {"system_preview": messages[0]["content"][:200], "user_preview": messages[1]["content"][:200]}
            })
            llm_t = time.perf_counter()
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    VERCEL_AI_GATEWAY_URL,
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "temperature": request.temperature,
                        "max_tokens": 2048,
                    },
                    headers={"Authorization": f"Bearer {gateway_key}", "Content-Type": "application/json"},
                    timeout=30.0,
                )
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"AI Gateway request failed: {resp.text}")
            llm_json = resp.json()
            response_text = llm_json["choices"][0]["message"]["content"]
            traces.append({
                "phase": "llm", "action": "gateway_request",
                "duration_ms": int((time.perf_counter()-llm_t)*1000),
                "meta": {"model": MODEL_NAME, "status_code": resp.status_code}
            })
            if isinstance(llm_json, dict) and isinstance(llm_json.get("usage"), dict):
                u = llm_json["usage"]
                traces.append({
                    "phase": "llm", "action": "usage", "duration_ms": 0,
                    "meta": {"tokens_prompt": u.get("prompt_tokens"), "tokens_completion": u.get("completion_tokens"), "tokens_total": u.get("total_tokens")}
                })
            traces.append({
                "phase": "llm", "action": "response_preview", "duration_ms": 0,
                "meta": {"preview": (response_text[:240] + "…") if len(response_text) > 240 else response_text}
            })

        traces.append({"phase": "done", "action": "total", "duration_ms": int((time.perf_counter()-t0)*1000)})
        return QueryResponse(query=request.query, response=response_text, model=MODEL_NAME, traces=traces)

    except HTTPException as e:
        # Return traces on error for better observability
        try:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail, "traces": traces})
        except Exception:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        # Include traces in generic errors as well
        try:
            return JSONResponse(status_code=500, content={"detail": f"Error: {str(e)}", "traces": traces})
        except Exception:
            return JSONResponse(status_code=500, content={"detail": f"Error: {str(e)}"})


# Entry point for local development
# Entry point for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

