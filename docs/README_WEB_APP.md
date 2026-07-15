# Web App - Local Development Setup

## Quick Start

### Easiest Method: Automated Launcher

From project root directory:
```cmd
start_web_app.bat
```

This starts both servers and opens the browser automatically.

### Manual Method

**Terminal 1 - Backend (port 5000):**
```cmd
cd <PROJECT_ROOT>
.venv\Scripts\activate
python docs\backend\app.py
```

**Terminal 2 - Frontend (port 8000):**
```cmd
cd <PROJECT_ROOT>
.venv\Scripts\activate
python start_frontend_server.py
```

**Browser:**
```
http://localhost:8000
```

## Why Two Servers?

### Backend Server (port 5000)
- Flask API for strad monitoring integration
- Provides REST endpoints
- Runs DL classification
- Connects to database

### Frontend Server (port 8000)
- Serves HTML, CSS, JS files
- Required for CORS to work properly
- Browser can't make API calls from file:// to http://

## Important Notes

❌ **Don't open `index.html` directly!**
- Opening `file:///path/to/index.html` causes CORS errors
- Browser blocks API calls from file:// to http://localhost:5000

✅ **Use the frontend server:**
- Serves files via HTTP: `http://localhost:8000`
- CORS works correctly
- Backend connection successful

## Architecture

```
Browser (http://localhost:8000)
    │
    ├─> Frontend Server (port 8000)
    │   └─> Serves: index.html, script.js, styles.css
    │
    └─> Backend API (port 5000)
        └─> Provides: /api/strads/recent, /api/inference, etc.
```

## Connection Status

**Green dot (●) = Connected**
- Both servers running
- Backend available
- Real data mode

**Red dot (○) = Disconnected**
- Backend not running or unavailable
- Demo mode (placeholder data)
- Videos still work

## Features

### Always Available (Frontend Only)
- Demo video playback
- Modal dialogs
- UI navigation
- Image upload interface

### Available When Connected (Backend + Frontend)
- Real data from database
- Live DL classification
- Snapshot retrieval
- Statistics API
- Connection status indicator

## Troubleshooting

### "Connection shows disconnected"
1. Check both servers are running
2. Backend: `http://localhost:5000` (check terminal)
3. Frontend: `http://localhost:8000` (check terminal)
4. Refresh browser (F5)

### "CORS error in console"
```
Access to fetch at 'http://localhost:5000' from origin 'file://' 
has been blocked by CORS policy
```

**Fix:** Don't open file directly! Use frontend server.

### "Port already in use"
```
OSError: [Errno 48] Address already in use
```

**Backend (port 5000):**
```cmd
netstat -ano | findstr :5000
taskkill /PID <process_id> /F
```

**Frontend (port 8000):**
```cmd
netstat -ano | findstr :8000
taskkill /PID <process_id> /F
```

## Testing

### 1. Check Servers Running

**Backend health check:**
```cmd
curl http://localhost:5000/
```

**Frontend health check:**
```
http://localhost:8000 (open in browser)
```

### 2. Test Features

1. Check connection status (top right)
2. Click "View Demo" on any card
3. Scroll to live inference test
4. Upload an image
5. Check results

### 3. Check Browser Console

Press F12 and look for:
```javascript
Backend connection: {status: 'running', ...}
Loaded recent strads: [...]
```

## Files

- `index.html` - Main web page
- `script.js` - JavaScript logic
- `styles.css` - Styling
- `backend/app.py` - Flask API server

## More Information

See project root documentation:
- `WEB_APP_QUICK_START.md` - Detailed web app guide
- `HOW_TO_USE_RIGHT_NOW.md` - Complete system guide
- `WEB_APP_INTEGRATION_SUMMARY.md` - Technical integration details
