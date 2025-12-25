CPSC 491 Movie Selector System. 


## 💻 Installation

### Step 1: download the zip file to desire location then cd to that location. 



### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

**Verification**: Your terminal should show `(venv)` prefix.

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt


pip install kagglehub

pip install nltk


# in addition, you will needs to install the dataset itself and include into the project folder. Link: 
https://drive.google.com/file/d/19HK_CQAZ3Yd6InNU3YQWgjZG1XpnvTax/view?usp=sharing


# run the training set, this a simpfly version, takes about 2 min
.\venv\Scripts\python.exe training\train_fast_dev.py
```



### Step 4: Database Setup

```bash
# Run database migrations
python manage.py migrate
```

**Output**: Should show migrations applied successfully.

### Step 5: Start Development Server

```bash
python manage.py runserver
```

**Output**: 
```
Starting development server at http://127.0.0.1:8000/
```

### Step 6: Verify Installation

Open your browser and navigate to:
```
http://localhost:8000
```



