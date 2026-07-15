# Deployment Guide: GitHub Pages Web Application

This document provides step-by-step instructions for deploying the Camera Misalignment Detection System web application to GitHub Pages.

---

## Prerequisites

Before deploying the web application, ensure you have:

### Required Accounts and Tools
- **GitHub account** - Free account at [github.com](https://github.com)
- **Git installed locally** - Download from [git-scm.com](https://git-scm.com)
- **Repository with docs/ directory** - The web application files must be in the `docs/` folder

### Repository Setup
1. **Create or use existing GitHub repository**
   - Repository can be public or private (both work with GitHub Pages)
   - Ensure you have write access to the repository

2. **Verify docs/ directory structure**
   ```
   docs/
   ├── index.html                   # Main HTML page ✓
   ├── styles.css                   # Styling ✓
   ├── script.js                    # JavaScript behavior ✓
   ├── 01_normal_operation.mp4      # Normal scenario video ✓
   ├── 01_normal_operation.gif      # Normal scenario fallback ✓
   ├── 02_impact_scenario.mp4       # Impact scenario video ✓
   └── 02_impact_scenario.gif       # Impact scenario fallback ✓
   ```

3. **Ensure all files are committed and pushed**
   ```bash
   git add docs/
   git commit -m "Add GitHub Pages web application"
   git push origin main
   ```

---

## Local Testing

**IMPORTANT**: Always test the application locally before deploying to GitHub Pages. This helps catch issues early and ensures a smooth deployment.

### Option 1: Python HTTP Server (Recommended)

Python's built-in HTTP server is the easiest way to test locally:

```bash
# Navigate to the docs directory
cd docs

# Python 3.x (most common)
python -m http.server 8000

# Python 2.x (if Python 3 not available)
python -m SimpleHTTPServer 8000
```

**Access the application**:
1. Open your browser
2. Navigate to: `http://localhost:8000`
3. Test all interactive features

**To stop the server**: Press `Ctrl+C` in the terminal

### Option 2: Live Server (VS Code Extension)

If you use Visual Studio Code, the Live Server extension provides automatic reloading:

**Installation**:
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X or Cmd+Shift+X)
3. Search for "Live Server" by Ritwick Dey
4. Click "Install"

**Usage**:
1. Open the `docs/index.html` file in VS Code
2. Right-click anywhere in the file
3. Select "Open with Live Server"
4. Browser opens automatically at `http://127.0.0.1:5500/docs/index.html`

**Features**:
- Automatic browser reload when you save changes
- Works with HTML, CSS, and JavaScript files
- No command-line required

### Option 3: Direct File Open (Not Recommended)

You can open `index.html` directly in your browser, but this has limitations:

```bash
# Windows
start docs/index.html

# macOS
open docs/index.html

# Linux
xdg-open docs/index.html
```

**Limitations**:
- Video files may fail to load due to CORS restrictions
- Some JavaScript features may not work correctly
- Does not accurately reflect how GitHub Pages will serve the site

**Verdict**: Only use this for quick HTML/CSS checks. Use a local server for proper testing.

### Local Testing Checklist

Before deploying, verify the following works locally:

- [ ] Page loads without errors (check browser console: F12)
- [ ] All sections visible: Header, Stats, Kanban Board, Timeline
- [ ] CSS styling applied correctly (purple gradient background, cards display properly)
- [ ] JavaScript is active (check browser console for errors)
- [ ] Clicking "View Demo" opens modal with video
- [ ] Video plays with native browser controls (play, pause, seek)
- [ ] Modal closes via:
  - Close button (X)
  - Clicking outside the modal (on the backdrop)
  - Pressing the Escape key
- [ ] All 5 scenario cards' "View Demo" buttons work
- [ ] "Details" button shows scenario information without video
- [ ] Hover effects work on cards and buttons
- [ ] Responsive design: Resize browser window to test mobile layout

---

## GitHub Pages Configuration

### Step 1: Enable GitHub Pages

1. **Navigate to your repository on GitHub**
   - Go to `https://github.com/[your-username]/[your-repository-name]`

2. **Open repository Settings**
   - Click the "Settings" tab (top navigation bar)
   - If you don't see "Settings", you may not have write access

3. **Navigate to Pages section**
   - In the left sidebar, scroll down to "Code and automation"
   - Click on "Pages"

4. **Configure source settings**
   - Under "Build and deployment" > "Source", select: **Deploy from a branch**
   - Under "Branch":
     - **Branch**: Select `main` (or `master` if that's your default branch)
     - **Folder**: Select `/docs`
   - Click **Save**

5. **Wait for deployment**
   - GitHub will display: "Your site is ready to be published at `https://[username].github.io/[repository-name]/`"
   - Initial deployment takes 1-2 minutes
   - Subsequent deployments are typically faster (30-60 seconds)

### Step 2: Monitor Deployment Status

**Check GitHub Actions**:
1. Go to the "Actions" tab in your repository
2. Look for a workflow named "pages build and deployment"
3. Green checkmark (✓) = successful deployment
4. Red X (✗) = deployment failed (click to view logs)

**Deployment Timeline**:
- **Queueing**: 5-10 seconds
- **Building**: 30-60 seconds
- **Deploying**: 10-20 seconds
- **Total**: 1-2 minutes

### Step 3: Access Your Live Site

Once deployment completes:

**URL Format**: `https://[username].github.io/[repository-name]/`

**Examples**:
- Username: `johndoe`, Repository: `experimenting`
  - URL: `https://johndoe.github.io/experimenting/`
- Username: `alice`, Repository: `camera-alignment`
  - URL: `https://alice.github.io/camera-alignment/`

**Important Notes**:
- URL is **case-sensitive** for the repository name
- If your repository name has capital letters, the URL will too
- HTTPS is enforced automatically by GitHub Pages
- Custom domains can be configured in Settings > Pages > Custom domain

### Step 4: Custom Domain (Optional)

To use a custom domain like `camera-demo.example.com`:

1. **In GitHub Settings > Pages**:
   - Under "Custom domain", enter your domain: `camera-demo.example.com`
   - Click "Save"
   - GitHub creates a `CNAME` file in your `docs/` directory

2. **In your DNS provider**:
   - Add a `CNAME` record:
     - **Name**: `camera-demo` (subdomain)
     - **Value**: `[username].github.io`
     - **TTL**: 3600 (or default)

3. **Wait for DNS propagation** (5 minutes to 24 hours)

4. **Enable HTTPS** (optional but recommended):
   - After DNS propagates, check "Enforce HTTPS" in GitHub Pages settings

---

## Verification Checklist

After deployment, verify everything works correctly on the live site:

### Page Load and Structure
- [ ] **Site accessible** - URL loads without 404 error
- [ ] **No broken assets** - Check browser console (F12) for 404 errors
- [ ] **Header displays** - Title, logo, and badges visible
- [ ] **Stats section visible** - 4 stat cards showing cameras, FPS, features, diamond
- [ ] **Kanban board renders** - 3 columns with scenario cards
- [ ] **Timeline section visible** - "Complete Impact Scenario Timeline" card

### Styling and Visual Design
- [ ] **CSS loaded correctly** - Purple gradient background visible
- [ ] **Cards styled properly** - White cards with shadows and rounded corners
- [ ] **Colors correct** - Green (normal), yellow (warning), red (critical) headers
- [ ] **Fonts render** - System fonts load (not broken/default serif)
- [ ] **Icons display** - Truck icon, emoji icons (📹, ⚡, 🎯, 💎)
- [ ] **Hover effects work** - Cards lift slightly on hover

### JavaScript Functionality
- [ ] **Buttons respond** - Cursor changes to pointer on buttons
- [ ] **Modal opens** - Clicking "View Demo" opens video modal
- [ ] **Video loads** - MP4 video appears in modal
- [ ] **Video plays** - Click play button, video starts
- [ ] **Video controls work** - Pause, seek, volume controls functional
- [ ] **Modal closes via close button** - Click X button, modal disappears
- [ ] **Modal closes via outside click** - Click backdrop, modal closes
- [ ] **Modal closes via Escape key** - Press Esc, modal closes
- [ ] **Video cleanup** - After closing modal, video stops playing

### All Scenarios Functional
Test each card's "View Demo" button:
- [ ] **Normal Operation** - Loads `01_normal_operation.mp4`
- [ ] **Camera 1: Minor Shift** - Loads `02_impact_scenario.mp4`
- [ ] **Camera 0: Wind Gust** - Loads `02_impact_scenario.mp4`
- [ ] **Camera 2: Debris Impact** - Loads `02_impact_scenario.mp4`
- [ ] **Camera 3: Strong Wind** - Loads `02_impact_scenario.mp4`
- [ ] **Complete Timeline** - Loads `02_impact_scenario.mp4` with full details

### Details Button
- [ ] **"Details" button works** - Opens modal without video
- [ ] **Scenario information displays** - Description and metrics visible
- [ ] **Timeline renders for impact scenarios** - Event list with frames

### Responsive Design
Test on different screen sizes (use browser DevTools):

- [ ] **Desktop (1920px)** - Full layout, 3 columns side-by-side
- [ ] **Desktop (1440px)** - Full layout, 3 columns
- [ ] **Laptop (1024px)** - Columns stack vertically
- [ ] **Tablet (768px)** - Stats cards stack, header reorganizes
- [ ] **Mobile (375px)** - Single column layout, readable text
- [ ] **Small mobile (320px)** - All content fits, no horizontal scroll

### Performance
- [ ] **Initial load < 3 seconds** - Page renders quickly
- [ ] **Modal opens quickly** - No lag when clicking "View Demo"
- [ ] **Smooth animations** - Fade-in and slide-up effects smooth
- [ ] **No layout shift** - Page doesn't jump during load
- [ ] **Scrolling smooth** - No lag or jank on mobile

### Cross-Browser Testing (Optional but Recommended)
- [ ] **Chrome** - Latest version
- [ ] **Firefox** - Latest version
- [ ] **Safari** - Latest version (macOS/iOS)
- [ ] **Edge** - Latest version
- [ ] **Mobile Safari** - iOS device
- [ ] **Chrome Mobile** - Android device

---

## Troubleshooting

### Issue: Page Shows 404 Error

**Symptoms**: Accessing GitHub Pages URL returns "404 - File not found"

**Possible Causes**:
1. GitHub Pages not enabled in repository settings
2. Wrong folder selected (should be `/docs`)
3. Files not in `docs/` directory
4. Branch not pushed to GitHub

**Solutions**:
1. **Verify GitHub Pages settings**:
   - Go to Settings > Pages
   - Ensure "Source" is set to branch `main` and folder `/docs`
   - Click "Save" again to refresh

2. **Check file location**:
   ```bash
   # Verify files are in docs/ directory
   git ls-tree -r main --name-only | grep "^docs/"
   ```
   - Should list: `docs/index.html`, `docs/styles.css`, `docs/script.js`, etc.

3. **Ensure branch is pushed**:
   ```bash
   git push origin main
   ```

4. **Wait for deployment**:
   - Check Actions tab for deployment status
   - Wait 1-2 minutes for deployment to complete

### Issue: CSS/JS Not Loading (Page Unstyled)

**Symptoms**: Page displays but has no styling, white background, default fonts

**Possible Causes**:
1. Incorrect file paths in HTML
2. Files not committed to repository
3. Case-sensitive file names on GitHub (vs local)

**Solutions**:
1. **Check browser console** (F12):
   - Look for 404 errors on `styles.css` or `script.js`
   - Note the exact URL that failed to load

2. **Verify file paths in index.html**:
   ```html
   <!-- Should be relative paths -->
   <link rel="stylesheet" href="styles.css">
   <script src="script.js"></script>
   
   <!-- NOT absolute paths -->
   <!-- <link rel="stylesheet" href="/docs/styles.css"> ❌ -->
   ```

3. **Ensure files are committed**:
   ```bash
   git status
   # Should show "working tree clean"
   
   # If files are untracked:
   git add docs/styles.css docs/script.js
   git commit -m "Add CSS and JavaScript files"
   git push origin main
   ```

4. **Check file name case**:
   - GitHub is case-sensitive: `Styles.css` ≠ `styles.css`
   - Ensure HTML references match exact file names

5. **Force refresh** browser cache:
   - Chrome/Edge: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
   - Firefox: `Ctrl+F5` (Windows) or `Cmd+Shift+R` (Mac)
   - Safari: `Cmd+Option+R` (Mac)

### Issue: Videos Not Playing

**Symptoms**: Modal opens but video doesn't load, black screen, or error message

**Possible Causes**:
1. Video files not in `docs/` directory
2. Large video files not uploaded (GitHub file size limits)
3. Video file names incorrect in `script.js`
4. Browser doesn't support MP4/H.264 codec

**Solutions**:
1. **Verify video files exist**:
   ```bash
   ls -lh docs/*.mp4 docs/*.gif
   # Should show 4 files: 2 MP4s + 2 GIFs
   ```

2. **Check video file sizes**:
   - GitHub has a 100 MB per-file limit
   - Current videos: 17 MB and 29 MB (well within limit)
   - If you see warnings during `git push`, files may be too large

3. **Use Git LFS for large files** (if needed):
   ```bash
   git lfs install
   git lfs track "*.mp4"
   git add .gitattributes
   git add docs/*.mp4
   git commit -m "Add videos with Git LFS"
   git push origin main
   ```

4. **Check browser console** (F12):
   - Look for error messages about video loading
   - Note exact error: 404, MIME type, codec, etc.

5. **Test GIF fallback**:
   - Temporarily rename MP4 file to trigger fallback
   - If GIF loads, problem is with MP4 codec support
   - Most browsers support H.264 MP4 (95%+ compatibility)

6. **Verify video file names in script.js**:
   ```javascript
   // Should match exact file names
   videoFile: '01_normal_operation.mp4'  // ✓
   // NOT:
   videoFile: '01_Normal_Operation.mp4'  // ❌ (wrong case)
   ```

### Issue: Modal Not Opening

**Symptoms**: Clicking "View Demo" does nothing, no modal appears

**Possible Causes**:
1. JavaScript file not loaded
2. JavaScript errors preventing execution
3. Browser console errors

**Solutions**:
1. **Open browser console** (F12):
   - Look for red error messages
   - Common errors:
     - `Uncaught ReferenceError: viewScenario is not defined`
     - `Uncaught SyntaxError: Unexpected token`

2. **Check if script.js loaded**:
   - Browser DevTools > Network tab
   - Look for `script.js` - should show status 200 (not 404)
   - If 404: script.js file missing or wrong path

3. **Verify JavaScript enabled**:
   - Some browsers/extensions block JavaScript
   - Check browser settings: JavaScript should be enabled

4. **Test in private/incognito mode**:
   - Browser extensions can interfere
   - Private mode disables most extensions

5. **Check onclick handlers in HTML**:
   ```html
   <!-- Should have onclick attribute -->
   <button class="btn btn-primary" onclick="viewScenario('normal')">
   ```

### Issue: Modal Doesn't Close

**Symptoms**: Modal opens but close button doesn't work, Escape key doesn't work

**Possible Causes**:
1. Event listeners not attached
2. JavaScript errors after modal opens

**Solutions**:
1. **Check browser console** for errors after opening modal

2. **Test all close methods**:
   - Click X button (top-right)
   - Click backdrop (dark area outside modal)
   - Press Escape key

3. **Verify closeModal function**:
   - Open console, type: `closeModal()`
   - If modal closes: event listeners not attached correctly
   - If error: JavaScript syntax issue

4. **Refresh page** and try again (clears any stuck state)

### Issue: Styling Broken on Mobile

**Symptoms**: Layout doesn't adjust on mobile, text too small, horizontal scrolling

**Possible Causes**:
1. Viewport meta tag missing
2. CSS media queries not loading
3. Browser zoom settings

**Solutions**:
1. **Verify viewport meta tag** in `index.html`:
   ```html
   <meta name="viewport" content="width=device-width, initial-scale=1.0">
   ```

2. **Check CSS media queries** in `styles.css`:
   ```css
   @media (max-width: 1024px) { /* ... */ }
   @media (max-width: 768px) { /* ... */ }
   ```

3. **Test in browser DevTools**:
   - F12 > Toggle device toolbar (Ctrl+Shift+M)
   - Select device: iPhone, iPad, etc.
   - Check if layout adapts

4. **Reset browser zoom**:
   - Chrome/Edge: `Ctrl+0` (Windows) or `Cmd+0` (Mac)
   - Zoom should be 100%

### Issue: Deployment Takes Too Long

**Symptoms**: GitHub Pages deployment stuck or exceeds 5 minutes

**Possible Causes**:
1. Repository too large
2. GitHub Actions queue backlog
3. GitHub service issues

**Solutions**:
1. **Check GitHub Actions**:
   - Actions tab > "pages build and deployment"
   - View logs for errors or warnings

2. **Check GitHub Status**:
   - Visit: [www.githubstatus.com](https://www.githubstatus.com)
   - Look for "GitHub Pages" incidents

3. **Cancel and re-trigger**:
   - Actions tab > Click workflow run > "Cancel workflow"
   - Make a small commit to re-trigger:
     ```bash
     git commit --allow-empty -m "Re-trigger deployment"
     git push origin main
     ```

4. **Repository size**:
   ```bash
   # Check repository size
   git count-objects -vH
   # If > 100 MB, consider cleaning large files
   ```

### Issue: "Permissions" Error in GitHub Actions

**Symptoms**: Deployment fails with "Resource not accessible by integration"

**Possible Causes**:
1. GitHub Actions permissions not enabled
2. Branch protection rules blocking deployment

**Solutions**:
1. **Enable GitHub Actions permissions**:
   - Settings > Actions > General
   - Under "Workflow permissions", select:
     - "Read and write permissions"
   - Check: "Allow GitHub Actions to create and approve pull requests"
   - Click "Save"

2. **Check branch protection**:
   - Settings > Branches
   - If branch protection enabled on `main`:
     - Allow GitHub Actions to bypass protection

3. **Use different branch**:
   - Create `gh-pages` branch and deploy from there instead

---

## Updating the Live Site

After deployment, any changes you push to the `main` branch and `docs/` directory will automatically trigger a new deployment.

### Workflow for Updates

1. **Make changes locally**:
   ```bash
   # Edit files in docs/ directory
   code docs/index.html  # or your preferred editor
   ```

2. **Test changes locally** (use Python HTTP server or Live Server)

3. **Commit changes**:
   ```bash
   git add docs/
   git commit -m "Update: description of changes"
   ```

4. **Push to GitHub**:
   ```bash
   git push origin main
   ```

5. **Monitor deployment**:
   - GitHub Actions tab > Watch deployment progress
   - Wait 1-2 minutes

6. **Verify changes live**:
   - Visit your GitHub Pages URL
   - Force refresh: `Ctrl+Shift+R` (or `Cmd+Shift+R` on Mac)

### Common Update Scenarios

**Updating content** (text, descriptions):
- Edit `index.html`
- No changes to CSS or JavaScript needed
- Fast deployment (~30 seconds)

**Updating styling** (colors, layout):
- Edit `styles.css`
- Test thoroughly on different screen sizes
- Clear browser cache after deployment

**Adding new scenarios**:
1. Edit `script.js` - Add new scenario to `SCENARIOS` object
2. Edit `index.html` - Add new scenario card
3. Add video files if needed
4. Test all interactive features locally first

**Replacing videos**:
1. Replace MP4/GIF files in `docs/` directory
2. Ensure file names match (or update `script.js`)
3. Check file sizes (GitHub: max 100 MB per file)
4. Test video playback locally before pushing

---

## Additional Resources

### GitHub Pages Documentation
- [Official GitHub Pages Guide](https://docs.github.com/en/pages)
- [Configuring a publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
- [Custom domains](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site)

### Testing Tools
- [W3C HTML Validator](https://validator.w3.org/) - Validate HTML
- [W3C CSS Validator](https://jigsaw.w3.org/css-validator/) - Validate CSS
- [Can I Use](https://caniuse.com/) - Check browser compatibility

### Browser DevTools
- [Chrome DevTools Guide](https://developer.chrome.com/docs/devtools/)
- [Firefox Developer Tools](https://firefox-source-docs.mozilla.org/devtools-user/)
- [Safari Web Inspector](https://developer.apple.com/safari/tools/)

### Git Resources
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
- [GitHub Guides](https://guides.github.com/)

---

## Support and Maintenance

### Getting Help

If you encounter issues not covered in this guide:

1. **Check browser console** (F12) for specific error messages
2. **Search GitHub Community**: [github.community](https://github.community)
3. **GitHub Support**: [support.github.com](https://support.github.com)
4. **Stack Overflow**: Tag questions with `github-pages`

### Monitoring Your Site

**GitHub provides deployment status**:
- Settings > Pages - Shows current deployment status
- Actions tab - Full deployment logs
- Insights > Traffic - Visitor statistics (14-day history)

**Setting up uptime monitoring** (optional):
- [UptimeRobot](https://uptimerobot.com/) - Free uptime monitoring
- [StatusCake](https://www.statuscake.com/) - Website monitoring
- [Pingdom](https://www.pingdom.com/) - Performance monitoring

### Keeping the Site Updated

**Regular maintenance tasks**:
- Update test status badge in header when system changes
- Replace demo videos when system capabilities improve
- Update scenario descriptions if detection logic changes
- Refresh browser testing on new browser versions

---

**Document Version**: 1.0  
**Last Updated**: 2026-06-19  
**System Version**: Phase 10 Complete (398/398 tests passing)
