# How to Upload This Project to GitHub

## Step-by-Step Guide

### Prerequisites
- Git installed on your computer
- GitHub account created
- Command line / Terminal access

---

## Method 1: Using GitHub Website (Easiest)

### Step 1: Create Repository on GitHub

1. Go to https://github.com
2. Click the **"+"** button (top right) → **"New repository"**
3. Fill in:
   - **Repository name**: `visitor-reid-counter` (or your choice)
   - **Description**: "Person Re-Identification system for counting unique visitors from camera footage"
   - **Public** or **Private**: Choose based on preference
   - **✅ Check**: "Add a README file" (we'll replace it)
   - Click **"Create repository"**

### Step 2: Upload Files

1. In your new repository, click **"Add file"** → **"Upload files"**
2. Drag and drop these files:
   - `memory_efficient_reid.py`
   - `reliability_checker.py`
   - `README.md`
   - `requirements.txt`
   - `.gitignore`
   - `LICENSE`
   - `METHODOLOGY.md`
3. Add commit message: "Initial commit - Visitor Re-ID system"
4. Click **"Commit changes"**

### Done! ✅

Your repository is now live at: `https://github.com/yourusername/visitor-reid-counter`

---

## Method 2: Using Git Command Line (More Professional)

### Step 1: Install Git

**Windows**: Download from https://git-scm.com/download/win
**Mac**: `brew install git` or download from https://git-scm.com/
**Linux**: `sudo apt-get install git`

### Step 2: Create Repository on GitHub

1. Go to https://github.com
2. Click **"+"** → **"New repository"**
3. Repository name: `visitor-reid-counter`
4. **DO NOT** check "Add a README file"
5. Click **"Create repository"**
6. **Copy the repository URL** (looks like: `https://github.com/yourusername/visitor-reid-counter.git`)

### Step 3: Prepare Your Local Folder

1. Put all your files in one folder:
```
visitor-reid-counter/
├── memory_efficient_reid.py
├── reliability_checker.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── METHODOLOGY.md
```

2. Open **Command Prompt** (Windows) or **Terminal** (Mac/Linux)
3. Navigate to your folder:
```bash
cd path/to/visitor-reid-counter
```

### Step 4: Initialize Git and Upload

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit - Visitor Re-ID system"

# Connect to GitHub (replace with YOUR repository URL)
git remote add origin https://github.com/yourusername/visitor-reid-counter.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 5: Enter GitHub Credentials

When prompted:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (not your password!)

**How to create token**:
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → Check "repo" scope
3. Copy the token and use it as password

### Done! ✅

---

## Method 3: Using GitHub Desktop (Visual Interface)

### Step 1: Install GitHub Desktop

Download from: https://desktop.github.com/

### Step 2: Sign In

1. Open GitHub Desktop
2. Sign in with your GitHub account

### Step 3: Create Repository

1. File → New Repository
2. Name: `visitor-reid-counter`
3. Local path: Choose where to save
4. Click "Create Repository"

### Step 4: Add Your Files

1. Copy all your Python files into the repository folder
2. GitHub Desktop will automatically detect them
3. Write commit message: "Initial commit"
4. Click "Commit to main"

### Step 5: Publish to GitHub

1. Click "Publish repository" button
2. Choose public/private
3. Click "Publish repository"

### Done! ✅

---

## After Uploading: Verify Your Repository

Visit: `https://github.com/yourusername/visitor-reid-counter`

You should see:
- ✅ All 7 files listed
- ✅ README.md displayed as homepage
- ✅ Green code button (for cloning)

---

## Updating Your Repository Later

If you make changes:

### Using GitHub Website:
1. Click on file
2. Click pencil icon (edit)
3. Make changes
4. Commit changes

### Using Command Line:
```bash
git add .
git commit -m "Description of changes"
git push
```

### Using GitHub Desktop:
1. Make changes to files
2. Commit in GitHub Desktop
3. Click "Push origin"

---

## Pro Tips

### 1. Don't Upload Large Files
The `.gitignore` file prevents uploading:
- Images (*.jpg, *.png)
- Generated reports (*.json)
- Result folders

**Why?** GitHub has 100MB file limit and repositories should be < 1GB

### 2. Write Good Commit Messages
```
✅ Good: "Add reliability checker with visual verification"
❌ Bad: "update"
```

### 3. Use Branches for Experiments
```bash
git checkout -b experiment
# make changes
git add .
git commit -m "Testing new threshold"
git push -u origin experiment
```

### 4. Add Screenshots
1. Create `images/` folder in repo
2. Add screenshots of results
3. Reference in README:
```markdown
![Results](images/results_screenshot.png)
```

---

## Common Issues & Solutions

### Issue: "Permission denied"
**Solution**: Make sure you're using Personal Access Token, not password

### Issue: "Large file detected"
**Solution**: Make sure `.gitignore` is working. Don't commit images!

### Issue: "Remote already exists"
**Solution**: 
```bash
git remote remove origin
git remote add origin YOUR_URL
```

### Issue: Files not showing on GitHub
**Solution**: Check you're on correct branch (should be "main")

---

## Customize Before Uploading

### 1. Update README.md
Replace placeholders:
- `[Your Name]` → Your actual name
- `[@yourusername]` → Your GitHub username
- `your.email@example.com` → Your email

### 2. Update LICENSE
Replace `[Your Name]` with your name

### 3. Remove Sensitive Data
Make sure no:
- Personal file paths
- Email addresses (if sensitive)
- Private information

---

## Want to Make It Professional?

### Add a Banner Image
Create `banner.png` and add to README:
```markdown
![Banner](banner.png)
```

### Add Badges
```markdown
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
```

### Add GitHub Actions
For automated testing (advanced)

---

## Need Help?

**GitHub Docs**: https://docs.github.com/
**Git Tutorial**: https://git-scm.com/book/en/v2

Feel free to ask if you get stuck! 🚀
