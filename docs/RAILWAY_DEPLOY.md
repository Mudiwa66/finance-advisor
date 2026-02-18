# Railway Deployment Guide

## Quick Deploy

1. **Create a Railway account** at [railway.app](https://railway.app)

2. **Deploy from GitHub**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose this repository

3. **Set Environment Variables** in Railway dashboard:

   **CRITICAL**: Environment variables MUST be set in Railway's dashboard, NOT in GitHub secrets or .env files!

   Steps to set environment variables in Railway:
   1. Go to your Railway project dashboard
   2. Click on your service/deployment
   3. Go to the "Variables" tab
   4. Click "+ New Variable" for each of the following:

   ```
   SUPABASE_URL=https://oanalylscevtqbriwnva.supabase.co
   SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9hbmFseWxzY2V2dHFpcml3bnZhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDcyNjMzNiwiZXhwIjoyMDg2MzAyMzM2fQ.7KhqD8shX1MouEV2JWZSsS7_K3psK8JFx23NaB6si48
   DEFAULT_USER_ID=c14571e0-874b-4d22-929e-f4e4fc73706b
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

   5. After adding all variables, Railway will automatically redeploy
   6. If it doesn't redeploy, click "Deploy" → "Redeploy"

4. **Optional Environment Variables**:
   ```
   GEMINI_MODEL=gemini-2.0-flash-preview
   FLASK_DEBUG=False
   ```

## Important Notes

- Railway automatically sets the `PORT` environment variable
- The app is configured to use `gunicorn` for production
- LLM uses Google Gemini API (gemini-2.0-flash-preview by default)
- Natural language queries are supported with the Gemini API key
- For Twilio webhook, use the Railway-provided URL: `https://your-app.railway.app/webhook`

## Post-Deployment

1. **Update Twilio Webhook**:
   - Go to Twilio Console → WhatsApp Sandbox
   - Update webhook URL to: `https://your-app.railway.app/webhook`

2. **Test the deployment**:
   - Access dashboard: `https://your-app.railway.app/dashboard`
   - Send a test WhatsApp message

3. **Configure ngrok (if needed)**:
   - Railway provides a public URL, so ngrok is no longer needed for production
   - Keep ngrok for local development only
