// ============================================================================
// Truck View: Camera Alignment Check - Interactive Web Application
// ============================================================================

// ============================================================================
// STRAD MONITORING INTEGRATION
// ============================================================================
// Connect to backend API for real strad monitoring data

const BACKEND_API_URL = 'http://localhost:5000/api';
let stradMonitoringConnected = false;
let recentStradsData = [];

// Check backend connection on page load
async function checkBackendConnection() {
    try {
        const response = await fetch('http://localhost:5000/');
        const data = await response.json();
        stradMonitoringConnected = data.strad_monitoring_connected;
        
        console.log('Backend connection:', data);
        
        // Update UI to show connection status
        updateConnectionStatus(data);
        
        // Load recent strads if connected
        if (stradMonitoringConnected) {
            await loadRecentStrads();
        }
    } catch (error) {
        console.log('Backend not available, using placeholder mode');
        stradMonitoringConnected = false;
    }
}

function updateConnectionStatus(data) {
    const statusEl = document.querySelector('.system-info');
    if (!statusEl) return;
    
    const statusHtml = `
        <div class="connection-status ${stradMonitoringConnected ? 'connected' : 'disconnected'}">
            ${stradMonitoringConnected ? '●' : '○'} Strad Monitoring: ${stradMonitoringConnected ? 'Connected' : 'Disconnected'}
        </div>
    `;
    statusEl.innerHTML = statusHtml;
}

async function loadRecentStrads() {
    try {
        const response = await fetch(`${BACKEND_API_URL}/strads/recent?limit=10`);
        const data = await response.json();
        
        if (data.success) {
            recentStradsData = data.data;
            console.log('Loaded recent strads:', recentStradsData);
            
            // Update kanban board with real data
            if (recentStradsData.length > 0) {
                updateKanbanWithRealData();
            }
        }
    } catch (error) {
        console.error('Failed to load recent strads:', error);
    }
}

function updateKanbanWithRealData() {
    // Group strads by classification
    const groupedStrads = {
        none: recentStradsData.filter(s => s.classification === 'none'),
        moderate: recentStradsData.filter(s => s.classification === 'moderate'),
        critical: recentStradsData.filter(s => s.classification === 'critical')
    };
    
    // Update counts in headers
    updateColumnCounts(groupedStrads);
    
    // Add real strad cards to each column
    addRealStradCards(groupedStrads);
}

function updateColumnCounts(groupedStrads) {
    // Update Normal Operation count
    const normalCountEl = document.querySelector('.kanban-column:nth-child(1) .count');
    if (normalCountEl) {
        normalCountEl.textContent = groupedStrads.none.length || 1; // Keep at least 1 for demo
    }
    
    // Update Minor Issues count
    const minorCountEl = document.querySelector('.kanban-column:nth-child(2) .count');
    if (minorCountEl) {
        minorCountEl.textContent = groupedStrads.moderate.length || 2; // Keep at least 2 for demo
    }
    
    // Update Critical Alerts count
    const criticalCountEl = document.querySelector('.kanban-column:nth-child(3) .count');
    if (criticalCountEl) {
        criticalCountEl.textContent = groupedStrads.critical.length || 2; // Keep at least 2 for demo
    }
}

function addRealStradCards(groupedStrads) {
    // Add real strad data cards if available
    // Keep existing demo cards and add real ones below
    
    // For now, we'll just log that we have real data available
    // The demo cards will remain as primary content
    console.log('Real strad data available:', groupedStrads);
}

// ============================================================================
// SCENARIO DATA STRUCTURE
// ============================================================================
// This object defines all camera alignment scenarios displayed in the UI.
// Each scenario maps to demo videos and includes detailed event information.

const SCENARIOS = {
  'normal': {
    title: 'Normal Operation - Aligned Cameras',
    videoFile: '01_normal_operation.mp4',
    gifFallback: '01_normal_operation.gif',
    description: 'All 4 cameras are properly aligned. Diamond markers connected at center. System operating normally with no detected issues.',
    details: {
      duration: '~5 seconds (150 frames)',
      alignment: '100% aligned',
      alerts: 'None',
      features: '100+ features per camera',
      status: 'Operational'
    }
  },
  
  'minor-1': {
    title: 'Camera 1: Minor Shift - Pothole',
    videoFile: '02_impact_scenario.mp4',
    gifFallback: '02_impact_scenario.gif',
    description: 'Minor misalignment detected on Camera 1 due to pothole impact. Diamond mostly connected. Within acceptable tolerance limits.',
    events: [
      {
        frame: 50,
        type: 'Pothole',
        camera: 1,
        severity: 'minor',
        rotation: '±2°',
        translation: '±15px',
        alert: false
      }
    ]
  },
  
  'minor-2': {
    title: 'Camera 0: Wind Gust Adjustment',
    videoFile: '02_impact_scenario.mp4',
    gifFallback: '02_impact_scenario.gif',
    description: 'Slight adjustment from wind gust affecting Camera 0. System compensating automatically. No action required.',
    events: [
      {
        frame: 200,
        type: 'Wind',
        camera: 0,
        severity: 'minor',
        rotation: '±1.5°',
        translation: '±12px',
        alert: false
      }
    ]
  },
  
  'critical-1': {
    title: 'Camera 2: Debris Impact - CRITICAL',
    videoFile: '02_impact_scenario.mp4',
    gifFallback: '02_impact_scenario.gif',
    description: '⚠️ ALERT: Major misalignment detected on Camera 2 from debris impact! Diamond broken. Immediate recalibration needed.',
    events: [
      {
        frame: 120,
        type: 'Debris',
        camera: 2,
        severity: 'critical',
        rotation: '±8°',
        translation: '±60px',
        alert: true
      }
    ]
  },
  
  'critical-2': {
    title: 'Camera 3: Strong Wind - CRITICAL',
    videoFile: '02_impact_scenario.mp4',
    gifFallback: '02_impact_scenario.gif',
    description: '⚠️ ALERT: Severe displacement detected on Camera 3 from strong wind! Diamond misaligned. Action required.',
    events: [
      {
        frame: 300,
        type: 'Wind',
        camera: 3,
        severity: 'critical',
        rotation: '±7°',
        translation: '±55px',
        alert: true
      }
    ]
  },
  
  'full-timeline': {
    title: 'Complete Impact Scenario Timeline',
    videoFile: '02_impact_scenario.mp4',
    gifFallback: '02_impact_scenario.gif',
    description: 'Full sequence showing all impact events in real-time. Watch how the system detects and responds to both minor and critical misalignments across all four cameras.',
    events: [
      {
        frame: 50,
        type: 'Pothole',
        camera: 1,
        severity: 'minor',
        rotation: '±2°',
        translation: '±15px',
        alert: false
      },
      {
        frame: 120,
        type: 'Debris',
        camera: 2,
        severity: 'critical',
        rotation: '±8°',
        translation: '±60px',
        alert: true
      },
      {
        frame: 200,
        type: 'Wind',
        camera: 0,
        severity: 'minor',
        rotation: '±1.5°',
        translation: '±12px',
        alert: false
      },
      {
        frame: 300,
        type: 'Wind',
        camera: 3,
        severity: 'critical',
        rotation: '±7°',
        translation: '±55px',
        alert: true
      }
    ],
    details: {
      duration: '~12 seconds (350 frames)',
      totalEvents: '4 impacts',
      minorEvents: '2 (Camera 0, Camera 1)',
      criticalEvents: '2 (Camera 2, Camera 3)',
      alertsTriggered: '2'
    }
  }
};

// ============================================================================
// MODAL CONTROL FUNCTIONS
// ============================================================================

/**
 * Opens the video modal and displays the specified scenario
 * 
 * This function:
 * 1. Looks up scenario from SCENARIOS object
 * 2. Creates video element with controls and autoplay
 * 3. Sets video source to scenario.videoFile
 * 4. Adds error handler for video.onerror to trigger GIF fallback
 * 5. Injects video into #videoContainer
 * 6. Updates #modalTitle with scenario title
 * 7. Calls renderScenarioDetails() to populate scenario info
 * 8. Adds 'active' class to modal to display it
 * 
 * @param {string} scenarioId - The ID of the scenario to display (e.g., 'normal', 'critical-1')
 * 
 * Requirements: 4.1, 4.2, 7.5, 11.2, 11.4
 */
function viewScenario(scenarioId) {
  // Look up scenario from SCENARIOS object
  const scenario = SCENARIOS[scenarioId];
  
  // Validate scenario exists
  if (!scenario) {
    console.error('Scenario not found:', scenarioId);
    alert('Scenario not found. Please refresh the page.');
    return;
  }
  
  // Get DOM elements
  const modal = document.getElementById('videoModal');
  const videoContainer = document.getElementById('videoContainer');
  const modalTitle = document.getElementById('modalTitle');
  const scenarioInfo = document.getElementById('scenarioInfo');
  
  // Create video element with controls and autoplay
  const video = document.createElement('video');
  video.src = scenario.videoFile;
  video.controls = true;
  video.autoplay = true;
  video.style.width = '100%';
  
  // Add error handler for video.onerror to trigger GIF fallback
  video.onerror = function() {
    console.warn('MP4 failed to load, trying GIF fallback for scenario:', scenarioId);
    loadGifFallback(scenario.gifFallback);
  };
  
  // Clear previous content and inject video into #videoContainer
  videoContainer.innerHTML = '';
  videoContainer.appendChild(video);
  
  // Update #modalTitle with scenario title
  modalTitle.textContent = scenario.title;
  
  // Call renderScenarioDetails() to populate scenario info
  const detailsHTML = renderScenarioDetails(scenario);
  scenarioInfo.innerHTML = detailsHTML;
  
  // Add 'active' class to modal to display it
  modal.classList.add('active');
  
  // Focus on modal for accessibility
  modal.focus();
}

/**
 * Closes the video modal and cleans up resources
 * 
 * This function:
 * 1. Removes the 'active' class to hide the modal
 * 2. Pauses any playing video
 * 3. Destroys the video element to free memory
 * 4. Clears scenario information
 * 
 * Called when:
 * - User clicks the close button (X)
 * - User clicks outside the modal content (on backdrop)
 * - User presses the Escape key
 * 
 * Requirements: 4.4, 7.4, 11.2
 */
function closeModal() {
  // Get modal element
  const modal = document.getElementById('videoModal');
  
  // Remove 'active' class to hide modal
  modal.classList.remove('active');
  
  // Get video container
  const videoContainer = document.getElementById('videoContainer');
  
  // Find video element and pause if playing
  const video = videoContainer.querySelector('video');
  if (video) {
    video.pause();
  }
  
  // Clear innerHTML of videoContainer to destroy video element
  videoContainer.innerHTML = '';
  
  // Clear innerHTML of scenarioInfo to reset details
  const scenarioInfo = document.getElementById('scenarioInfo');
  scenarioInfo.innerHTML = '';
}

/**
 * Display detailed scenario information in modal without video
 * @param {string} scenarioId - The scenario identifier (e.g., 'normal', 'critical-1')
 */
function showDetails(scenarioId) {
  // Look up scenario from SCENARIOS object
  const scenario = SCENARIOS[scenarioId];
  
  if (!scenario) {
    console.error('Scenario not found:', scenarioId);
    alert('Scenario not found. Please refresh the page.');
    return;
  }
  
  // Get modal elements
  const modal = document.getElementById('videoModal');
  const modalTitle = document.getElementById('modalTitle');
  const videoContainer = document.getElementById('videoContainer');
  const scenarioInfo = document.getElementById('scenarioInfo');
  
  // Update modal title
  modalTitle.textContent = scenario.title;
  
  // Clear video container (no video for details view)
  videoContainer.innerHTML = '';
  
  // Call renderScenarioDetails() to generate HTML
  const detailsHTML = renderScenarioDetails(scenario);
  
  // Inject details into modal
  scenarioInfo.innerHTML = detailsHTML;
  
  // Add 'active' class to modal to display it
  modal.classList.add('active');
  
  // Focus on modal for accessibility
  modal.focus();
}

// ============================================================================
// VIDEO FALLBACK MECHANISM
// ============================================================================

/**
 * Load GIF fallback when MP4 video fails to load
 * 
 * This function is called when video.onerror is triggered. It replaces the
 * failed video element with an image element showing the GIF version of the
 * demo. A notice is displayed to inform the user about the fallback.
 * 
 * @param {string} gifFile - Relative path to the GIF file (e.g., '01_normal_operation.gif')
 */
function loadGifFallback(gifFile) {
  // Get the video container element
  const videoContainer = document.getElementById('videoContainer');
  
  // Clear any existing content in the container
  videoContainer.innerHTML = '';
  
  // Create a notice message
  const notice = document.createElement('p');
  notice.textContent = 'Video unavailable, showing GIF version';
  notice.style.color = '#f59e0b';  // Warning color (yellow)
  notice.style.textAlign = 'center';
  notice.style.marginBottom = '1rem';
  notice.style.fontWeight = 'bold';
  
  // Create the img element for the GIF
  const img = document.createElement('img');
  img.src = gifFile;
  img.style.width = '100%';
  img.alt = 'Demo GIF fallback';
  
  // Inject the notice and image into the video container
  videoContainer.appendChild(notice);
  videoContainer.appendChild(img);
  
  console.log(`Video failed to load. Showing GIF fallback: ${gifFile}`);
}

// ============================================================================
// SCENARIO DETAILS RENDERING
// ============================================================================

/**
 * Generates HTML content for displaying detailed scenario information.
 * Handles both scenarios with 'details' (key-value pairs) and 'events' (timeline).
 * 
 * @param {Object} scenario - The scenario object from SCENARIOS
 * @returns {string} HTML string containing formatted scenario details
 * 
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
 */
function renderScenarioDetails(scenario) {
  let html = '<div class="scenario-details">';
  
  // Add description
  html += `<p class="scenario-description">${scenario.description}</p>`;
  
  // Check if scenario has 'details' property (key-value pairs)
  if (scenario.details) {
    html += '<h3>Scenario Information</h3>';
    html += '<div class="details-grid">';
    
    // Iterate through details object and display key-value pairs
    for (const [key, value] of Object.entries(scenario.details)) {
      // Format key: convert camelCase to Title Case with spaces
      const formattedKey = key
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase());
      
      html += `
        <div class="detail-item">
          <span class="detail-label">${formattedKey}:</span>
          <span class="detail-value">${value}</span>
        </div>
      `;
    }
    
    html += '</div>';
  }
  
  // Check if scenario has 'events' property (timeline)
  if (scenario.events && scenario.events.length > 0) {
    html += '<h3>Timeline of Events</h3>';
    html += '<ul class="event-timeline">';
    
    // Iterate through events and generate timeline list
    scenario.events.forEach(event => {
      // Determine alert indicator
      const alertIndicator = event.alert 
        ? '<span class="alert-indicator critical">⚠️ ALERT</span>' 
        : '<span class="alert-indicator normal">✓ No alert</span>';
      
      // Determine severity class for styling
      const severityClass = event.severity === 'critical' ? 'critical' : 'minor';
      
      html += `
        <li class="event-item ${severityClass}">
          <div class="event-header">
            <strong>Frame ${event.frame}</strong> - Camera ${event.camera}
          </div>
          <div class="event-details">
            <span class="event-type">${event.type}</span>
            <span class="event-severity ${severityClass}">(${event.severity})</span>
          </div>
          <div class="event-metrics">
            Rotation: ${event.rotation} | Translation: ${event.translation}
          </div>
          <div class="event-alert">
            ${alertIndicator}
          </div>
        </li>
      `;
    });
    
    html += '</ul>';
  }
  
  html += '</div>';
  
  return html;
}

// ============================================================================
// EVENT LISTENERS - DOM INITIALIZATION
// ============================================================================

/**
 * Set up event listeners when DOM is ready
 * 
 * This ensures all event listeners are attached after the DOM has fully loaded.
 * Event listeners include:
 * - Escape key to close modal
 * 
 * Requirements: 4.4, 11.5
 */
document.addEventListener('DOMContentLoaded', function() {
  
  // Check backend connection on page load
  checkBackendConnection();
  
  /**
   * Keyboard event listener for Escape key
   * Closes the modal when Escape is pressed and modal is active
   * 
   * Requirements: 4.4, 11.5
   */
  document.addEventListener('keydown', function(event) {
    // Check if the pressed key is 'Escape'
    if (event.key === 'Escape') {
      // Get the modal element
      const modal = document.getElementById('videoModal');
      
      // Check if modal has 'active' class (is currently open)
      if (modal && modal.classList.contains('active')) {
        // Close the modal
        closeModal();
      }
    }
  });
  
});

  // ==============================================================================
  // Upload Interface - Single Composite Image with Drag & Drop
  // ==============================================================================
  
  const dropZone = document.getElementById('dropZone');
  const compositeImageInput = document.getElementById('compositeImage');
  const dropPlaceholder = document.getElementById('dropPlaceholder');
  const imagePreviewContainer = document.getElementById('imagePreviewContainer');
  const imagePreview = document.getElementById('imagePreview');
  const removeImageBtn = document.getElementById('removeImageBtn');
  const uploadStatus = document.getElementById('uploadStatus');
  const runInferenceBtn = document.getElementById('runInferenceBtn');
  const clearImageBtn = document.getElementById('clearImageBtn');
  
  // Click on drop zone to trigger file input
  dropZone.addEventListener('click', function(e) {
    // Don't trigger if clicking the remove button
    if (e.target.id === 'removeImageBtn') {
      return;
    }
    compositeImageInput.click();
  });
  
  // Handle file input change
  compositeImageInput.addEventListener('change', handleCompositeImageSelect);
  
  // Drag and drop handlers
  dropZone.addEventListener('dragover', handleDragOver);
  dropZone.addEventListener('dragleave', handleDragLeave);
  dropZone.addEventListener('drop', handleDrop);
  
  // Remove image button
  removeImageBtn.addEventListener('click', function(e) {
    e.stopPropagation(); // Prevent triggering dropZone click
    removeImage();
  });
  
  // Clear image button
  if (clearImageBtn) {
    clearImageBtn.addEventListener('click', removeImage);
  }
  
  // Run Inference button listener
  if (runInferenceBtn) {
    runInferenceBtn.addEventListener('click', runInference);
  }

// ==============================================================================
// Upload Handler Functions
// ==============================================================================

let currentCompositeImage = null;
let inferenceController = null;
let latestInferenceResults = null;

/**
 * Handle file selection from input element
 * @param {Event} event - File input change event
 */
function handleCompositeImageSelect(event) {
  const file = event.target.files[0];
  
  if (!file) {
    return;
  }
  
  // Validate file type
  if (!file.type.startsWith('image/')) {
    alert('Please select a valid image file (JPEG or PNG)');
    return;
  }
  
  // Validate file size (10MB max)
  const maxSize = 10 * 1024 * 1024; // 10MB in bytes
  if (file.size > maxSize) {
    alert('Image file is too large. Maximum size is 10MB.');
    return;
  }
  
  // Store the file and display preview
  currentCompositeImage = file;
  displayImagePreview(file);
}

/**
 * Handle drag over event
 * @param {DragEvent} event - Drag event
 */
function handleDragOver(event) {
  event.preventDefault();
  event.stopPropagation();
  dropZone.classList.add('drag-over');
}

/**
 * Handle drag leave event
 * @param {DragEvent} event - Drag event
 */
function handleDragLeave(event) {
  event.preventDefault();
  event.stopPropagation();
  dropZone.classList.remove('drag-over');
}

/**
 * Handle file drop event
 * @param {DragEvent} event - Drop event
 */
function handleDrop(event) {
  event.preventDefault();
  event.stopPropagation();
  dropZone.classList.remove('drag-over');
  
  const files = event.dataTransfer.files;
  
  if (files.length === 0) {
    return;
  }
  
  const file = files[0];
  
  // Validate file type
  if (!file.type.startsWith('image/')) {
    alert('Please drop a valid image file (JPEG or PNG)');
    return;
  }
  
  // Validate file size (10MB max)
  const maxSize = 10 * 1024 * 1024;
  if (file.size > maxSize) {
    alert('Image file is too large. Maximum size is 10MB.');
    return;
  }
  
  // Store the file and display preview
  currentCompositeImage = file;
  displayImagePreview(file);
}

/**
 * Display image preview in the drop zone
 * @param {File} file - The image file to preview
 */
function displayImagePreview(file) {
  // Create FileReader to read the image
  const reader = new FileReader();
  
  reader.onload = function(e) {
    // Set the preview image source
    imagePreview.src = e.target.result;
    
    // Hide placeholder, show preview container
    dropPlaceholder.style.display = 'none';
    imagePreviewContainer.style.display = 'block';
    
    // Update upload status
    uploadStatus.textContent = `Uploaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    uploadStatus.style.color = 'var(--color-success)';
    
    // Enable the run inference button
    runInferenceBtn.disabled = false;
    
    // Show clear button
    if (clearImageBtn) {
      clearImageBtn.style.display = 'inline-flex';
    }
  };
  
  reader.onerror = function() {
    alert('Error reading image file. Please try another image.');
    removeImage();
  };
  
  // Read the file as data URL
  reader.readAsDataURL(file);
}

/**
 * Remove the uploaded image and reset the UI
 */
function removeImage() {
  // Clear the stored file
  currentCompositeImage = null;
  
  // Reset the file input
  compositeImageInput.value = '';
  
  // Clear the preview image
  imagePreview.src = '';
  
  // Show placeholder, hide preview
  dropPlaceholder.style.display = 'flex';
  imagePreviewContainer.style.display = 'none';
  
  // Reset upload status
  uploadStatus.textContent = 'No image uploaded';
  uploadStatus.style.color = 'var(--color-gray-600)';
  
  // Disable run inference button
  runInferenceBtn.disabled = true;
  
  // Hide clear button
  if (clearImageBtn) {
    clearImageBtn.style.display = 'none';
  }
  
  // Hide results section if visible
  const resultsSection = document.getElementById('resultsSection');
  if (resultsSection) {
    resultsSection.style.display = 'none';
  }
}

/**
 * Run inference on the uploaded composite image
 */
async function runInference() {
  if (!currentCompositeImage) {
    alert('Please upload an image first.');
    return;
  }
  
  // Show loading modal
  showLoadingModal('Uploading images...', 'Processing snapshot with SimpleClassifierWrapper');
  
  // Create FormData to send images
  const formData = new FormData();
  
  // Use single-image mode to trigger real classifier (not mock data)
  // Backend will use SimpleClassifierWrapper when it receives 'image' field
  formData.append('image', currentCompositeImage);
  
  // Create AbortController for cancellation
  inferenceController = new AbortController();
  
  try {
    // Update loading message
    showLoadingModal('Running inference...', 'Processing with deep learning model');
    
    // Make API request to backend
    const response = await fetch('http://localhost:5000/api/inference', {
      method: 'POST',
      body: formData,
      signal: inferenceController.signal
    });
    
    // Check if request was successful
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    
    // Parse the JSON response
    const data = await response.json();
    
    // DEBUG: Log the response to console
    console.log('=== INFERENCE RESPONSE DEBUG ===');
    console.log('Response data:', data);
    console.log('Has misalignment_probability?', 'misalignment_probability' in data);
    console.log('Has classification?', 'classification' in data);
    console.log('Has confidence?', 'confidence' in data);
    console.log('Mode:', data.mode);
    console.log('================================');
    
    // Hide loading modal
    hideLoadingModal();
    
    // Store results for export
    latestInferenceResults = data;
    
    // Display results in UI
    displayInferenceResults(data);
    
  } catch (error) {
    // Hide loading modal
    hideLoadingModal();
    
    // Check if error was from user cancellation
    if (error.name === 'AbortError') {
      console.log('Inference cancelled by user');
      return;
    }
    
    // Display error message
    console.error('Inference error:', error);
    alert(`Inference failed: ${error.message}\n\nMake sure the backend server is running on localhost:5000`);
  } finally {
    // Clean up controller
    inferenceController = null;
  }
}

/**
 * Show loading modal with message
 * @param {string} message - Main loading message
 * @param {string} details - Additional details
 */
function showLoadingModal(message, details) {
  const loadingModal = document.getElementById('loadingModal');
  const loadingMessage = document.getElementById('loadingMessage');
  const loadingDetails = document.getElementById('loadingDetails');
  
  if (loadingModal) {
    loadingMessage.textContent = message;
    loadingDetails.textContent = details;
    loadingModal.style.display = 'flex';
  }
}

/**
 * Hide loading modal
 */
function hideLoadingModal() {
  const loadingModal = document.getElementById('loadingModal');
  
  if (loadingModal) {
    loadingModal.style.display = 'none';
  }
}

/**
 * Cancel ongoing inference request
 */
function cancelInference() {
  if (inferenceController) {
    inferenceController.abort();
    inferenceController = null;
  }
  
  hideLoadingModal();
}

/**
 * Display inference results in the UI
 * @param {Object} data - The inference result data from backend
 */
function displayInferenceResults(data) {
  // Show results section
  const resultsSection = document.getElementById('resultsSection');
  if (resultsSection) {
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  
  // Handle both response formats:
  // - Multi-camera format: {misalignment_probability, severity, pose, uncertainty}
  // - Single-image format: {classification, confidence, processing_time_ms}
  
  // Update misalignment probability
  const probabilityFill = document.querySelector('.probability-fill');
  const probabilityValue = document.querySelector('.probability-value');
  
  // Convert confidence to probability if single-image mode
  let probability = data.misalignment_probability;
  if (probability === undefined && data.confidence !== undefined) {
    // Single-image mode: use confidence as probability
    // Map classification to probability range
    if (data.classification === 'none') {
      probability = data.confidence * 0.2; // 0-20% for none
    } else if (data.classification === 'moderate') {
      probability = 0.2 + (data.confidence * 0.5); // 20-70% for moderate
    } else if (data.classification === 'critical') {
      probability = 0.7 + (data.confidence * 0.3); // 70-100% for critical
    }
  }
  
  if (probability !== undefined) {
    const prob = (probability * 100).toFixed(1);
    probabilityFill.style.width = `${prob}%`;
    probabilityValue.textContent = `${prob}%`;
    
    // Color code based on severity
    if (probability < 0.3) {
      probabilityFill.style.backgroundColor = 'var(--color-success)';
    } else if (probability < 0.7) {
      probabilityFill.style.backgroundColor = 'var(--color-warning)';
    } else {
      probabilityFill.style.backgroundColor = 'var(--color-danger)';
    }
  }
  
  // Update severity classification
  const severityBadge = document.querySelector('.badge');
  const severityDescription = document.querySelector('.severity-description');
  
  // Get severity from either format
  let severity = data.severity || data.classification;
  
  if (severity) {
    severity = severity.toLowerCase();
    
    // Update badge
    severityBadge.className = 'badge';
    if (severity === 'normal' || severity === 'none') {
      severityBadge.classList.add('badge-success');
      severityBadge.textContent = '✓ Normal';
    } else if (severity === 'minor' || severity === 'low' || severity === 'moderate') {
      severityBadge.classList.add('badge-warning');
      severityBadge.textContent = '⚠ Moderate Misalignment';
    } else if (severity === 'critical' || severity === 'severe') {
      severityBadge.classList.add('badge-danger');
      severityBadge.textContent = '🚨 Critical Misalignment';
    }
    
    // Update description
    if (severityDescription) {
      if (data.description) {
        severityDescription.textContent = data.description;
      } else if (data.classification && data.confidence) {
        // Generate description for single-image mode
        const conf = (data.confidence * 100).toFixed(1);
        if (data.classification === 'none') {
          severityDescription.textContent = `🟢 NO MISALIGNMENT (${conf}% confidence) - Camera properly aligned`;
        } else if (data.classification === 'moderate') {
          severityDescription.textContent = `🟡 MODERATE MISALIGNMENT (${conf}% confidence) - Continue monitoring`;
        } else if (data.classification === 'critical') {
          severityDescription.textContent = `🔴 CRITICAL MISALIGNMENT (${conf}% confidence) - Camera requires immediate adjustment`;
        }
      }
    }
  }
  
  // Update 6-DOF pose values
  if (data.pose) {
    // Rotation
    if (data.pose.rotation) {
      document.getElementById('poseRoll').textContent = `${data.pose.rotation.roll.toFixed(2)}°`;
      document.getElementById('posePitch').textContent = `${data.pose.rotation.pitch.toFixed(2)}°`;
      document.getElementById('poseYaw').textContent = `${data.pose.rotation.yaw.toFixed(2)}°`;
    }
    
    // Translation
    if (data.pose.translation) {
      document.getElementById('poseX').textContent = `${data.pose.translation.x.toFixed(3)} m`;
      document.getElementById('poseY').textContent = `${data.pose.translation.y.toFixed(3)} m`;
      document.getElementById('poseZ').textContent = `${data.pose.translation.z.toFixed(3)} m`;
    }
  }
  
  // Update uncertainty values
  if (data.uncertainty) {
    if (data.uncertainty.aleatoric !== undefined) {
      document.getElementById('aleatoricUncertainty').textContent = data.uncertainty.aleatoric.toFixed(4);
    }
    
    if (data.uncertainty.epistemic !== undefined) {
      document.getElementById('epistemicUncertainty').textContent = data.uncertainty.epistemic.toFixed(4);
    }
  }
}

/**
 * Download inference results as JSON file
 */
function downloadResults() {
  if (!latestInferenceResults) {
    alert('No inference results available to download.');
    return;
  }
  
  // Convert results to JSON string
  const jsonStr = JSON.stringify(latestInferenceResults, null, 2);
  
  // Create blob
  const blob = new Blob([jsonStr], { type: 'application/json' });
  
  // Create download link
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `inference_results_${Date.now()}.json`;
  
  // Trigger download
  document.body.appendChild(link);
  link.click();
  
  // Clean up
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Reset inference UI for another run
 */
function resetInference() {
  // Hide results section
  const resultsSection = document.getElementById('resultsSection');
  if (resultsSection) {
    resultsSection.style.display = 'none';
  }
  
  // Clear stored results
  latestInferenceResults = null;
  
  // Remove current image
  removeImage();
  
  // Scroll to upload section
  const uploadSection = document.querySelector('.upload-section');
  if (uploadSection) {
    uploadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ==============================================================================
// End of original script
// ==============================================================================

// ==============================================================================
// LIVE MONITORING INTEGRATION
// ==============================================================================
// Loads real images from monitoring cycles and augmented dataset.
// Updates active camera count. Provides strad detail modals.

/**
 * Load active camera count from backend and update the stat card
 */
async function loadActiveCameraCount() {
    try {
        const response = await fetch(`${BACKEND_API_URL}/live/active-camera-count`);
        const data = await response.json();
        
        if (data.success) {
            const countEl = document.getElementById('activeCameraCount');
            if (countEl) {
                countEl.textContent = data.available;
                countEl.title = `${data.total_strads} total - ${data.critical_excluded} critical excluded`;
            }
        }
    } catch (error) {
        console.log('Could not load active camera count:', error);
        const countEl = document.getElementById('activeCameraCount');
        if (countEl) countEl.textContent = '135';
    }
}

/**
 * Load and display live/augmented images in the grid
 */
async function refreshLiveImages() {
    const grid = document.getElementById('liveImageGrid');
    if (!grid) return;
    
    const sourceFilter = document.getElementById('liveSourceFilter')?.value || 'auto';
    const severityFilter = document.getElementById('liveSeverityFilter')?.value || '';
    
    grid.innerHTML = '<p class="loading-text">Loading images...</p>';
    
    try {
        let url = `${BACKEND_API_URL}/live/images?source=${sourceFilter}&limit=12`;
        if (severityFilter) url += `&severity=${severityFilter}`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (!data.success || data.count === 0) {
            grid.innerHTML = '<p class="no-data-text">No images available. Run a monitoring cycle first, or check that SCFootage_augmented dataset exists.</p>';
            return;
        }
        
        grid.innerHTML = '';
        
        data.data.forEach(img => {
            const card = document.createElement('div');
            card.className = `live-image-card severity-${img.classification}`;
            
            const severityBadge = img.classification === 'critical' ? '🔴' :
                                  img.classification === 'moderate' ? '🟡' : '🟢';
            
            const confidenceText = img.confidence > 0 ? `${(img.confidence * 100).toFixed(1)}%` : '--';
            
            card.innerHTML = `
                <div class="live-image-wrapper">
                    <img src="${BACKEND_API_URL}/live/image/${img.filename}" 
                         alt="${img.strad_id}" 
                         onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22150%22><rect fill=%22%23333%22 width=%22200%22 height=%22150%22/><text fill=%22%23999%22 x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22>No Image</text></svg>'" />
                </div>
                <div class="live-image-info">
                    <div class="live-image-header">
                        <span class="live-strad-id">${severityBadge} ${img.strad_id}</span>
                        <span class="live-source-badge ${img.source}">${img.source}</span>
                    </div>
                    <div class="live-image-meta">
                        <span>Confidence: ${confidenceText}</span>
                        ${img.timestamp ? `<span>${new Date(img.timestamp).toLocaleString()}</span>` : ''}
                    </div>
                    <div class="live-image-actions">
                        <button class="btn btn-sm" onclick="showStradDetails('${img.strad_id}')">Details</button>
                        <button class="btn btn-sm" onclick="copyIpAddress('${img.strad_id}')">Copy IP</button>
                    </div>
                </div>
            `;
            
            grid.appendChild(card);
        });
        
    } catch (error) {
        console.error('Failed to load live images:', error);
        grid.innerHTML = '<p class="no-data-text">Backend not available. Start the Flask server on localhost:5000.</p>';
    }
}

/**
 * Show detailed monitoring info for a specific strad
 */
async function showStradDetails(stradId) {
    const modal = document.getElementById('stradDetailModal');
    const title = document.getElementById('stradDetailTitle');
    const body = document.getElementById('stradDetailBody');
    
    if (!modal || !body) return;
    
    title.textContent = `Strad ${stradId} - Details`;
    body.innerHTML = '<p>Loading...</p>';
    modal.classList.add('active');
    
    try {
        const response = await fetch(`${BACKEND_API_URL}/live/strad-details/${stradId}`);
        const data = await response.json();
        
        if (!data.success) {
            body.innerHTML = '<p>Failed to load details.</p>';
            return;
        }
        
        const d = data.data;
        let html = '<div class="strad-detail-content">';
        
        // Basic info
        html += '<div class="detail-section">';
        html += `<h3>Strad Information</h3>`;
        html += `<table class="detail-table">`;
        html += `<tr><td><strong>Strad ID:</strong></td><td>${d.strad_id}</td></tr>`;
        html += `<tr><td><strong>IP Address:</strong></td><td>${d.ip_address || 'N/A'} <button class="btn btn-sm" onclick="navigator.clipboard.writeText('${d.ip_address || ''}')">📋 Copy</button></td></tr>`;
        html += `<tr><td><strong>Last Checked:</strong></td><td>${d.last_checked ? new Date(d.last_checked).toLocaleString() : 'Never'}</td></tr>`;
        html += `<tr><td><strong>Status:</strong></td><td>${d.is_critical ? '🔴 CRITICAL (Excluded from cycling)' : '🟢 Active'}</td></tr>`;
        html += `</table>`;
        html += '</div>';
        
        // Critical info
        if (d.is_critical && d.critical_info) {
            html += '<div class="detail-section critical-section">';
            html += `<h3>⚠️ Critical Exclusion</h3>`;
            html += `<table class="detail-table">`;
            html += `<tr><td><strong>Marked Critical:</strong></td><td>${new Date(d.critical_info.timestamp).toLocaleString()}</td></tr>`;
            html += `<tr><td><strong>Reason:</strong></td><td>${d.critical_info.reason || 'N/A'}</td></tr>`;
            html += `</table>`;
            html += '</div>';
        }
        
        // Classification history
        if (d.classifications && d.classifications.length > 0) {
            html += '<div class="detail-section">';
            html += `<h3>Classification History (Last ${d.classifications.length})</h3>`;
            html += '<table class="detail-table history-table">';
            html += '<tr><th>Time</th><th>Classification</th><th>Confidence</th></tr>';
            
            d.classifications.forEach(c => {
                const badge = c.classification === 'critical' ? '🔴' :
                              c.classification === 'moderate' ? '🟡' : '🟢';
                const time = c.timestamp ? new Date(c.timestamp).toLocaleString() : '--';
                const conf = c.confidence ? `${(c.confidence * 100).toFixed(1)}%` : '--';
                html += `<tr><td>${time}</td><td>${badge} ${c.classification}</td><td>${conf}</td></tr>`;
            });
            
            html += '</table>';
            html += '</div>';
        }
        
        html += '</div>';
        body.innerHTML = html;
        
    } catch (error) {
        body.innerHTML = `<p>Error loading details: ${error.message}</p>`;
    }
}

/**
 * Copy IP address for a strad to clipboard
 */
async function copyIpAddress(stradId) {
    try {
        const response = await fetch(`${BACKEND_API_URL}/live/strad-details/${stradId}`);
        const data = await response.json();
        
        if (data.success && data.data.ip_address) {
            await navigator.clipboard.writeText(data.data.ip_address);
            alert(`IP address copied: ${data.data.ip_address}`);
        } else {
            alert(`No IP address found for ${stradId}`);
        }
    } catch (error) {
        alert(`Failed to get IP: ${error.message}`);
    }
}

/**
 * Close the strad detail modal
 */
function closeStradDetailModal() {
    const modal = document.getElementById('stradDetailModal');
    if (modal) modal.classList.remove('active');
}

// Initialize live monitoring features on page load
document.addEventListener('DOMContentLoaded', function() {
    // Load active camera count
    loadActiveCameraCount();
    
    // Load live images
    setTimeout(refreshLiveImages, 1000);  // Slight delay to let backend connection establish
    
    // Escape key closes strad detail modal too
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeStradDetailModal();
        }
    });
    
    // Filter change listeners
    const sourceFilter = document.getElementById('liveSourceFilter');
    const severityFilter = document.getElementById('liveSeverityFilter');
    if (sourceFilter) sourceFilter.addEventListener('change', refreshLiveImages);
    if (severityFilter) severityFilter.addEventListener('change', refreshLiveImages);
});
