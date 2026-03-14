# Deploy checklist – fix “nothing has deployed”

Follow these steps in order. If a step fails, check the section at the end.

---

## Step 1: Push code to GitHub

From your project folder (where `Api.py` is):

```powershell
git add -A
git status
git commit -m "Unified backend: Api.py, DB, RAG, deploy config"
git remote add origin https://github.com/Yash-povie/updated-Nanotoxi-test.git
# If origin already exists: git remote set-url origin https://github.com/Yash-povie/updated-Nanotoxi-test.git
git push -u origin main
```

If the remote has different history:

```powershell
git pull origin main --allow-unrelated-histories --no-edit
git push origin main
```

Confirm on GitHub that the latest commit and files (e.g. `Api.py`, `Procfile`, `railway.json`) are on the repo.

---

## Step 2: Create / connect Railway project

1. Go to [railway.app](https://railway.app) and sign in.
2. **New Project** → **Deploy from GitHub repo**.
3. Choose **Yash-povie/updated-Nanotoxi-test**.
4. Select the **main** branch (or the branch you pushed to).
5. Railway will create a **service** and start a build.

---

## Step 3: Add Postgres and set DATABASE_URL

1. In the same project, click **New** → **Database** → **PostgreSQL** (or use an existing Postgres service).
2. Open the **Postgres** service → **Variables** or **Connect** → copy the **Connection URL** (e.g. `postgresql://postgres:xxx@xxx.railway.app:5432/railway`).
3. Open the **web service** (the one that runs your code) → **Variables** → **New Variable**:
   - Name: `DATABASE_URL`
   - Value: paste the Postgres connection URL.
4. Click **Redeploy** (or wait for the next deploy) so the app restarts with `DATABASE_URL`.

---

## Step 4: Check deploy and logs

1. **Build**
   - In the web service, open **Deployments** → latest deployment.
   - Build must finish with **Success**. If it fails, open the build logs:
     - **Missing module**: install in `requirements.txt` (e.g. `fastapi`, `uvicorn`, `psycopg2-binary`, `PyJWT`).
     - **Wrong Python**: set in **Variables** or add `runtime.txt` (e.g. `python-3.11.0`).

2. **Run**
   - After build, the **start command** runs: `uvicorn Api:app --host 0.0.0.0 --port $PORT`.
   - In **Deployments** → your deploy → **View logs**:
     - You should see something like: `[startup] DATABASE_URL set: True` and `Database initialized successfully.`
     - If you see **Application failed to respond** or crashes, read the logs (e.g. `ModuleNotFoundError: No module named 'Api'` → file must be `Api.py` in the root).

3. **URL**
   - Web service → **Settings** → **Networking** → **Generate domain** (or use the default).
   - Open `https://your-app.up.railway.app/health` in a browser → should return `{"status":"ok",...}`.

---

## Step 5: Test the API

- Health: `GET https://your-app.up.railway.app/health`
- Docs: `https://your-app.up.railway.app/docs`
- Login: `POST https://your-app.up.railway.app/auth/token` with body `{"username":"admin","password":"titan2024"}`
- Then use the returned token for `/predict`, `/dataset/info`, etc.

---

## If “nothing has deployed” or build fails

| Symptom | What to do |
|--------|-------------|
| No deploy at all | Confirm repo is connected (Step 2). Check correct branch. Trigger **Redeploy** in Railway. |
| Build failed | Open **Build logs** in the deployment. Fix missing deps in `requirements.txt` or Python version. |
| Start / crash after build | Open **Deploy logs**. Fix missing env (e.g. `DATABASE_URL`) or module (e.g. `Api`). |
| 502 / “Application failed to respond” | App may be crashing. Check deploy logs; ensure `Api:app` starts and binds to `0.0.0.0:$PORT`. |
| Database empty in cloud | Set `DATABASE_URL` on the **web** service (Step 3) and **Redeploy**. |

---

## Files Railway uses

- **Procfile** or **railway.json** `startCommand`: how the app is started.
- **requirements.txt**: Python dependencies.
- **Api.py**: must be in the **root** of the repo (same level as `Procfile`).

After you push, Railway rebuilds from the branch you selected. Every new push to that branch can trigger a new deploy.
