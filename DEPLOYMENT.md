# Deployment Guide

## 📋 Pre-Deployment Checklist

### ✅ Files to Include in GitHub
- [ ] All Python code (`app/`, `ml_model/`)
- [ ] Templates (`app/templates/*.html`)
- [ ] Static files (`app/static/`)
- [ ] Requirements (`requirements.txt`)
- [ ] Config files (`vercel.json`, `wsgi.py`, `runtime.txt`)
- [ ] Documentation (`README.md`, `.env.example`)
- [ ] ML models (if < 100MB each, otherwise use Git LFS)

### ❌ Files to NEVER Push (Already in .gitignore)
- [ ] `.env` - Contains your API keys!
- [ ] `venv/` - Virtual environment
- [ ] `__pycache__/` - Python cache files
- [ ] `*.pyc` - Compiled Python files
- [ ] `.vscode/`, `.idea/` - IDE settings
- [ ] `*.log` - Log files

## 🚀 GitHub Setup

### Step 1: Initialize Git (if not already done)
```bash
cd e:\Life
git init
git add .
git commit -m "Initial commit: LifePulse health assessment platform"
```

### Step 2: Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `lifepulse` (or your choice)
3. Description: "AI-powered health assessment platform"
4. Choose Public or Private
5. **Do NOT initialize with README** (we already have one)
6. Click "Create repository"

### Step 3: Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/lifepulse.git
git branch -M main
git push -u origin main
```

### Step 4: Verify
- Check that `.env` is NOT visible on GitHub
- Verify all other files are present
- Check that `.gitignore` is working

## 🌐 Vercel Deployment

### Method 1: Vercel Dashboard (Recommended)

1. **Sign up/Login to Vercel**
   - Go to https://vercel.com
   - Sign up with GitHub account

2. **Import Project**
   - Click "Add New..." → "Project"
   - Import your GitHub repository
   - Vercel will auto-detect Flask

3. **Configure Build Settings**
   - Framework Preset: Other
   - Build Command: (leave empty)
   - Output Directory: (leave empty)
   - Install Command: `pip install -r requirements.txt`

4. **Add Environment Variables**
   Click "Environment Variables" and add:
   ```
   SECRET_KEY = your-secret-key-here
   GEMINI_API_KEY = your-gemini-api-key-here
   FLASK_ENV = production
   ```

5. **Deploy**
   - Click "Deploy"
   - Wait for build to complete (2-5 minutes)
   - Get your deployment URL: `https://your-project.vercel.app`

### Method 2: Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel

# Add environment variables
vercel env add SECRET_KEY
vercel env add GEMINI_API_KEY
vercel env add FLASK_ENV

# Deploy to production
vercel --prod
```

## ⚠️ Important Deployment Notes

### ML Models
Your ML models might be large files. Check sizes:
```bash
# Check model sizes
du -sh app/models/heart/*.pkl
du -sh app/models/sleep/*.pkl
du -sh saved_*/*.pkl
```

**If models > 100MB:**
1. Use Git LFS (Large File Storage)
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git add .gitattributes
   git commit -m "Add Git LFS"
   ```

2. Or host models externally (AWS S3, Google Cloud Storage)

### Vercel Limitations
- **100MB** deployment size limit
- **50MB** per serverless function
- **10 second** function timeout (Hobby plan)

**If you hit limits:**
- Consider deploying to Render, Railway, or Heroku instead
- Or upgrade to Vercel Pro
- Or optimize/compress models

## 🔄 Alternative Deployment Platforms

### Railway
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Render
1. Go to https://render.com
2. Create Web Service
3. Connect GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn wsgi:app`
6. Add environment variables

### Heroku
```bash
# Install Heroku CLI
# Create Procfile
echo "web: gunicorn wsgi:app" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
heroku config:set SECRET_KEY=your-key
heroku config:set GEMINI_API_KEY=your-key
```

## 🧪 Testing After Deployment

1. **Test all routes:**
   - Home page: `/`
   - Health Calculator: `/health`
   - Heart Disease: `/heart_disease`
   - Sleep Disorder: `/sleep`
   - Migraine: `/migraine`
   - Health Score: `/health-score`
   - Nutrition: `/nutrition`

2. **Test ML predictions:**
   - Submit forms with sample data
   - Verify predictions work
   - Check AI advice generates

3. **Check errors:**
   - Vercel Dashboard → Logs
   - Look for missing dependencies
   - Check API key issues

## 🐛 Common Issues

### Issue: "Module not found"
**Solution:** Add missing package to `requirements.txt`

### Issue: "API key not found"
**Solution:** Check environment variables in Vercel dashboard

### Issue: "Function timeout"
**Solution:** ML model loading might be slow
- Cache models
- Or use smaller models
- Or upgrade plan

### Issue: "Deployment size too large"
**Solution:** 
- Remove unnecessary files
- Use Git LFS for models
- Or deploy to Railway/Render

## 📊 Monitoring

### Vercel Analytics
- Enable in Project Settings → Analytics
- Track visits, performance, Web Vitals

### Error Tracking
- Check Vercel Logs regularly
- Set up Sentry for error tracking (optional)

## 🔒 Security Best Practices

1. **Never commit `.env` file**
2. **Use strong SECRET_KEY** (generate with `python -c "import secrets; print(secrets.token_hex(32))"`)
3. **Keep dependencies updated** (`pip list --outdated`)
4. **Set HTTPS only** (Vercel does this automatically)
5. **Add rate limiting** (optional, for production)

## 📈 Post-Deployment

1. **Update README.md** with live URL
2. **Add badges** (deployment status, etc.)
3. **Set up custom domain** (optional)
4. **Monitor performance**
5. **Gather user feedback**

---

Need help? Open an issue on GitHub or check Vercel documentation.
