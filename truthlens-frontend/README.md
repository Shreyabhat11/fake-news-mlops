# TruthLens — Frontend

A React + TypeScript + Vite + Tailwind frontend for the **Fake News Trend Drift Detector**
FastAPI backend. This replaces the project's Streamlit UI. It does not touch the backend,
the trained model, or the prediction logic — it only calls the existing API.

Backend (unchanged): `https://fake-news-mlops.onrender.com`

## Project structure

```
truthlens-frontend/
├── src/
│   ├── components/       UI components (Navbar, Hero, AnalyzerCard, ResultPanel, …)
│   ├── hooks/            useHealth, useAnalyzer, useModelInfo
│   ├── services/api.ts   All backend communication lives here
│   ├── types/api.ts      TypeScript types mirroring the FastAPI Pydantic schemas
│   ├── lib/               constants.ts (examples, static model facts), cn.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── index.html
├── vite.config.ts
├── tailwind.config.js
├── package.json
└── .env.example
```

## Local development

```bash
npm install
cp .env.example .env    # points at the deployed Render API by default
npm run dev
```

Opens at `http://localhost:5173`.

## Production build

```bash
npm run build      # outputs to dist/
npm run preview    # serve the production build locally
```

Deploy `dist/` to any static host (Vercel, Netlify, GitHub Pages, Render static site, etc.),
with `VITE_API_URL` set as an environment variable at build time if you don't want the
default backend URL.

## How the frontend talks to the API

All requests go through `src/services/api.ts`, which reads the base URL from
`VITE_API_URL` (falls back to the deployed Render URL if unset). No component calls
`fetch` directly.

- `GET /health` — polled every 20s (`useHealth`) to drive the status indicator in the nav bar.
- `GET /model/info` — fetched once on load to populate live model version/F1 in the Model section.
- `POST /predict` — called from the "Analyze article" button.
- `POST /batch_predict` — called from the collapsible Batch analysis section.

**Render cold starts:** the free-tier backend spins down when idle and can take up to a
minute to wake. `/predict` and `/batch_predict` use a 60s timeout; if a response hasn't
come back within ~2.5s, the UI switches its loading copy to "starting up" rather than
looking frozen. `/health` uses a short 8s timeout instead, so the status indicator doesn't
block the page waiting on a cold instance — a fast timeout there just shows "waking" state
and retries on the next poll.

Errors are categorized (`network`, `timeout`, `validation`, `rate_limited`, `server`,
`unknown`) in `types/api.ts` / `services/api.ts` and mapped to plain-language messages in
`ResultPanel.tsx` — no raw stack traces or JSON are ever shown to the user.

## What's static vs. live in the Model section

The API's `/model/info` endpoint returns `model_version`, `classifier`, `val_f1`, and
`test_f1` — those are fetched live. It does **not** return dataset size, ROC-AUC, or
embedding dimension, so those three figures (`44,856 articles`, `0.984` ROC-AUC, `384`
dimensions) are hardcoded in `src/lib/constants.ts` from the project's training report.
If you retrain and those numbers change, update `MODEL_FACTS` there.

## Known gaps / things to verify yourself before treating this as done

I built and manually reviewed this in a sandboxed environment with **no network access**,
so I could not:

- Run `npm install` / `npm run build` / `npm run dev` to confirm it actually compiles
- Hit `https://fake-news-mlops.onrender.com` from the browser to confirm CORS, response
  shapes, and cold-start behavior match what's coded here
- Check the production build in an actual browser, console errors included, or test the
  responsive layout on a real device

I did do a manual pass for: Tailwind class validity (caught and fixed `h-4.5`/`w-4.5`,
which aren't on Tailwind's default spacing scale), the missing `@` path alias in
`vite.config.ts` (fixed), brace/paren balance across every file, and cross-checked prop
types between components by hand. That's a substitute for compiling, not a replacement —
please run:

```bash
npm install
npm run build
npm run dev
```

and open the browser console while testing the real Analyze flow against the live Render
backend before relying on this for a demo. If `npm run build` throws TypeScript errors,
they're most likely small — mismatched prop names or a missed import — and should be fast
to fix from the compiler output.

## Removing the old Streamlit frontend

`app.py` (the Streamlit app) is no longer used by this frontend. It's left untouched in
the repo since removing it wasn't done here — delete it yourself once you've confirmed
this frontend covers everything you need, or keep it around as an internal ops dashboard
if the batch/drift-monitoring views in it are still useful to you.
