import wandb
E, C, P = "akareiva4-denver-university", "taxi-fare-model", "comp4450-taxi-fare"
cands = [
    f"{E}/wandb-registry-model/{C}:production",
    f"wandb-registry-model/{C}:production",
    f"{E}/model-registry/{C}:production",
    f"{E}/{P}/{C}:production",
    f"{E}/{P}/{C}:latest",
]
api = wandb.Api()
ok = []
for r in cands:
    try:
        a = api.artifact(r, type="model")
        print(f"WORKS   {r}  (version={a.version})")
        ok.append(r)
    except Exception as e:
        print(f"fails   {r}\n        {str(e).splitlines()[0][:90]}")
print("\nWANDB_MODEL_REF=" + ok[0] if ok else "\nNone worked - use MODEL_SOURCE=local")
