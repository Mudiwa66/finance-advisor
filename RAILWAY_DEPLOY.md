# Railway Deployment Guide

## Quick Deploy

1. **Create a Railway account** at [railway.app](https://railway.app)

2. **Deploy from GitHub**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose this repository

3. **Set Environment Variables** in Railway dashboard:
   ```
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_service_role_key
   DEFAULT_USER_ID=your_default_user_uuid
   ```

4. **Optional Environment Variables**:
   ```
   OLLAMA_URL=http://your-ollama-instance/api/chat
   OLLAMA_MODEL=llama3.2
   FLASK_DEBUG=False
   ```

## Important Notes

- Railway automatically sets the `PORT` environment variable
- The app is configured to use `gunicorn` for production
- Ollama must be deployed separately (or use a hosted instance)
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
