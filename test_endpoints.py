import json
import time
from urllib import request, error


BASE = "http://127.0.0.1:8000"


def do(method: str, path: str, body=None, token: str | None = None):
    url = BASE + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw
    except Exception as e:
        return None, str(e)


def main() -> None:
    # Give server a moment to be fully ready if just started
    time.sleep(1.5)

    print("HEALTH:")
    print(do("GET", "/health"))

    print("\nAUTH admin:")
    code, admin_token_resp = do(
        "POST", "/auth/token", {"username": "admin", "password": "titan2024"}
    )
    print(code, admin_token_resp)
    admin_token = (
        admin_token_resp.get("access_token") if isinstance(admin_token_resp, dict) else None
    )

    print("\nAUTH customer:")
    code, cust_token_resp = do(
        "POST", "/auth/token", {"username": "researcher1", "password": "nano@123"}
    )
    print(code, cust_token_resp)
    customer_token = (
        cust_token_resp.get("access_token") if isinstance(cust_token_resp, dict) else None
    )

    print("\nPREDICT (no RAG) as customer:")
    code, resp = do(
        "POST",
        "/predict",
        {
            "composition": "ZnO",
            "nanoparticle_size": 50.0,
            "zeta_potential": -25.0,
            "morphology": "spherical",
            "cell_type": "A549",
            "dosage_in_vitro": 10.0,
            "organic_inorganic": "inorganic",
            "surface_chemistry": "PEG",
            "top_k": 3,
            "use_rag": False,
            "generate_report": False,
        },
        token=customer_token,
    )
    print(code, resp)

    print("\nPREDICT (with RAG) as customer:")
    code, resp = do(
        "POST",
        "/predict",
        {
            "composition": "ZnO",
            "nanoparticle_size": 50.0,
            "zeta_potential": -25.0,
            "morphology": "spherical",
            "cell_type": "A549",
            "dosage_in_vitro": 10.0,
            "organic_inorganic": "inorganic",
            "surface_chemistry": "PEG",
            "top_k": 2,
            "use_rag": True,
            "generate_report": False,
        },
        token=customer_token,
    )
    print(code, str(resp)[:400])

    print("\nADMIN dataset/info:")
    print(do("GET", "/dataset/info", token=admin_token))

    print("\nADMIN logs:")
    print(do("GET", "/logs", token=admin_token))

    print("\nADMIN dataset/info with customer (should be 403):")
    print(do("GET", "/dataset/info", token=customer_token))

    print("\nRAG info:")
    print(do("GET", "/rag/info", token=customer_token))

    print("\nRAG search:")
    print(
        do(
            "POST",
            "/rag/search",
            {"query": "What is toxicity of ZnO?", "top_k": 2},
            token=customer_token,
        )
    )

    print("\nRAG answer:")
    print(
        do(
            "POST",
            "/rag/answer",
            {"question": "Explain toxicity thresholds for ZnO category", "top_k": 2},
            token=customer_token,
        )
    )


if __name__ == "__main__":
    main()

