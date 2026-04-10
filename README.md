# Ergon Contract Intelligence — Frontend

React frontend for the Ergon Contract Intelligence agent.

## Setup

```bash
npm install
```

Create a `.env.local` file:
```
REACT_APP_API_URL=https://your-api-id.execute-api.us-east-1.amazonaws.com/query
```

Replace the URL with your actual API Gateway endpoint.

## Run locally

```bash
npm start
```

## Deploy to Vercel

1. Push this folder to a GitHub repo
2. Go to vercel.com → New Project → import the repo
3. Add environment variable: `REACT_APP_API_URL` = your API Gateway URL
4. Deploy

Vercel will auto-detect the React app and build it.
