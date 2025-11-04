# 🚀 Quick Deployment Guide

## 📦 What to Push to GitHub

### ✅ INCLUDE These:
```
✓ app/                         (All application code)
✓ ml_model/                    (Training scripts)
✓ requirements.txt             (Dependencies)
✓ run.py                       (Local dev server)
✓ wsgi.py                      (Production server)
✓ vercel.json                  (Vercel config)
✓ runtime.txt                  (Python version)
✓ README.md                    (Documentation)
✓ .env.example                 (Template for env vars)
✓ .gitignore                   (Already configured!)
✓ ML models in app/models/     (If < 100MB)
✓ Static files (CSS, JS, images)
✓ Templates (HTML files)
```

### ❌ NEVER Push These (in .gitignore):
```
✗ .env                         (CONTAINS API KEYS!)
✗ venv/                        (Virtual environment - 100s of MB)
✗ __pycache__/                 (Python cache)
✗ *.pyc, *.pyo                 (Compiled Python)
✗ .vscode/, .idea/             (IDE settings)
✗ *.log                        (Log files)
✗ Life.code-workspace          (VS Code workspace)
```

---

## 🔐 Sensitive Data to Hide

### Your .env file contains:
- `SECRET_KEY` - Flask session encryption
- `GEMINI_API_KEY` - Google AI API key

**⚠️ CRITICAL:** These MUST stay in `.env` and NEVER be pushed to GitHub!

---

## 🎯 Step-by-Step: Push to GitHub

### 1. Check .gitignore is working
```bash
# See what will be committed
git status

# Make sure .env is NOT listed!
# Make sure venv/ is NOT listed!
```

### 2. Initialize and Push
```bash
# Initialize git (if not done)
git init

# Add all files (respects .gitignore)
git add .

# Commit
git commit -m "Initial commit: LifePulse health platform"

# Create GitHub repo at https://github.com/new
# Then link and push:
git remote add origin https://github.com/YOUR_USERNAME/lifepulse.git
git branch -M main
git push -u origin main
```

### 3. Verify on GitHub
- ✅ Check that code is visible
- ❌ Verify `.env` is NOT there
- ❌ Verify `venv/` is NOT there

---

## 🌐 Deploy to Vercel

### Option A: Vercel Dashboard (Easiest)

1. **Go to:** https://vercel.com
2. **Sign up** with your GitHub account
3. **Click:** "Add New..." → "Project"
4. **Import** your repository
5. **Configure:**
   - Framework: Other
   - Root Directory: `./`
   - Build Command: (leave empty)
   - Output Directory: (leave empty)

6. **Environment Variables** (CRITICAL!):
   Click "Environment Variables" tab and add:
   ```
   Name: SECRET_KEY
   Value: [Generate a new one - see below]
   
   Name: GEMINI_API_KEY
   Value: [Your Gemini API key]
   
   Name: FLASK_ENV
   Value: production
   ```

7. **Deploy!** Click deploy and wait 2-5 minutes

### Option B: Vercel CLI
```bash
npm i -g vercel
vercel login
vercel
# Follow prompts, add env vars when asked
vercel --prod
```

---

## 🔑 Generate New SECRET_KEY

**Don't use your local dev key in production!**

```python
# Run this in Python:
import secrets
print(secrets.token_hex(32))
```

Copy the output and use as `SECRET_KEY` in Vercel.

---

## 📊 Get Gemini API Key

1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key
4. Add to Vercel environment variables

---

## ⚠️ Important Notes

### Model File Sizes
Check if your ML models are too large:
```bash
# Windows
dir app\models\heart\*.pkl
dir app\models\sleep\*.pkl

# See sizes of all pkl files
```

**If models > 100MB:**
- Vercel might fail to deploy
- Consider using Git LFS
- Or deploy to Railway/Render instead

### Vercel Free Tier Limits
- ✅ Good for: Most Flask apps
- ⚠️ Limits: 100MB deployment, 50MB functions, 10s timeout
- 💡 Alternative: Railway, Render (better for ML apps)

---

## 🧪 After Deployment

### Test Everything:
```
✓ Home page loads
✓ All navigation links work
✓ Health calculator functions
✓ Heart disease prediction works
✓ Sleep disorder prediction works
✓ Migraine prediction works
✓ Health score calculation works
✓ Nutrition search works
✓ AI advice generates (Gemini API working)
```

### Check Logs:
- Vercel Dashboard → Your Project → Logs
- Look for errors
- Check if ML models loaded

---

## 🐛 Troubleshooting

### "Module not found" error
→ Add the package to `requirements.txt` and redeploy

### "GEMINI_API_KEY not found"
→ Check environment variables in Vercel dashboard

### "Deployment too large"
→ Models might be too big. Try Railway or Render instead.

### App is slow
→ ML model loading takes time. Consider:
- Using smaller models
- Caching predictions
- Upgrading to Vercel Pro

---

## 📱 Share Your App

Once deployed, you'll get a URL like:
```
https://lifepulse-xyz123.vercel.app
```

Share it and test from different devices!

---

## 🎉 You're Ready!

Your checklist:
- [ ] `.gitignore` created ✓
- [ ] `.env` is in .gitignore ✓
- [ ] `.env.example` created ✓
- [ ] All config files created ✓
- [ ] Pushed to GitHub (without .env!)
- [ ] Deployed to Vercel
- [ ] Added environment variables to Vercel
- [ ] Tested all features

**Need help?** Check `DEPLOYMENT.md` for detailed guide!
