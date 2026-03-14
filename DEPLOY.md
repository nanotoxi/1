# Deploy this backend to the cloud first, then use it directly

Deploy the single backend (Api.py) to Railway so you have one URL. Your frontend and any client then call that URL; no need to run the backend locally.

## 1. Deploy to Railway

1. **Push this repo to GitHub** (if not already).
2. In **Railway** → **New Project** → **Deploy from GitHub repo** → select this repo.
3. Railway will use:
   - **Procfile:** `web: uvicorn Api:app --host 0.0.0.0 --port ${PORT:-8000}`
   - **railway.json** (if present): same idea, healthcheck on `/health`.

4. Add **Postgres** in the same project if you don’t have it:  
   **New** → **Database** → **PostgreSQL**.  
   Copy the **Connection URL** from the Postgres service (Variables or Connect).

5. In your **web service** (the one that runs the API):
   - **Variables** → **New Variable**
   - `DATABASE_URL` = paste the Postgres connection URL (from step 4).
   - Optional: `JWT_SECRET` = a long random string for production.

6. **Redeploy** the web service (or wait for the next deploy).  
   After deploy, open the **generated URL** (e.g. `https://your-app.up.railway.app`).

## 2. Use the deployed backend directly

- **API base URL:** `https://your-app.up.railway.app`
- **Docs:** `https://your-app.up.railway.app/docs`
- **Health:** `https://your-app.up.railway.app/health`

Point your frontend (or Postman, curl) at this URL:

- `POST https://your-app.up.railway.app/auth/token` → get JWT
- `POST https://your-app.up.railway.app/predict` with `Authorization: Bearer <token>`
- `GET https://your-app.up.railway.app/rag/info`, etc.

All predictions and logs are stored in the same Railway Postgres because `DATABASE_URL` is set in the cloud.

## 3. Checklist

| Step | Done |
|------|------|
| Repo deployed to Railway (web service) | |
| Postgres added in same project | |
| `DATABASE_URL` set in web service Variables | |
| Redeploy after adding variables | |
| Open `/health` in browser → `{"status":"ok"}` | |
| Call `/auth/token` then `/predict` → rows appear in DB | |

## 4. Optional: custom domain

In Railway → your web service → **Settings** → **Domains** → add a custom domain if you want (e.g. `api.yourdomain.com`).
