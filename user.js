// This will capture the image from the webcam 
document.getElementById('captureBtn').addEventListener('click', async function() {
    const video = document.getElementById('cameraStream');
    const canvas = document.getElementById('snapshot');
    const context = canvas.getContext('2d');
    
    // Set canvas size to video size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw the video frame onto the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
  
    // Convert canvas to image file
    canvas.toBlob(async function(blob) {
      const formData = new FormData();
      formData.append('image', blob, 'snapshot.jpg');  // Ensure the field name is 'image'
      
      // Send image to backend
      const response = await fetch('/upload_image', {  // Match endpoint
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (response.ok) {
        // Show result
        document.getElementById('categoryResult').innerText = data.category;  // Match 'category'
        document.getElementById('confidenceResult').innerText = (data.confidence * 100).toFixed(2) + '%';

        // Dynamically add the image to the classification table
        const imgElement = document.createElement('img');
        imgElement.src = data.image_url; // Use the image URL from the response
        imgElement.alt = 'Uploaded Image';
        imgElement.width = 100; // Adjust size as needed
        imgElement.height = 100; // Adjust size as needed
        
        // Ensure the columns are properly populated
        const historyTable = document.getElementById('historyTable');
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${data.id}</td>
          <td>${data.category}</td>
          <td>${data.confidence}</td>
          <td>${data.timestamp}</td>
          <td><img src="${data.image_url}" width="100" height="100" alt="Uploaded Image"></td>
        `;
        historyTable.appendChild(row);

        // Refresh stats and history dynamically
        populateStatistics();
        populateHistory();
      } else {
        alert('Error: ' + data.error);
      }
    });
});

// Handle image upload from file input
document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent form from reloading the page
    
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    
    if (!file) {
      alert('Please upload an image');
      return;
    }
  
    const formData = new FormData();
    formData.append('image', file);
  
    // Show loading spinner
    document.getElementById('loadingSpinner').classList.remove('d-none');
    
    // Send image to backend for classification
    const response = await fetch('/upload_image', {
      method: 'POST',
      body: formData
    });
  
    const data = await response.json();
    
    // Hide loading spinner
    document.getElementById('loadingSpinner').classList.add('d-none');
    
    if (response.ok) {
      // Show result
      document.getElementById('categoryResult').innerText = data.category;
      document.getElementById('confidenceResult').innerText = (data.confidence * 100).toFixed(2) + '%';

      // Dynamically add the image to the classification table
      const imgElement = document.createElement('img');
      imgElement.src = data.image_url; // Use the image URL from the response
      imgElement.alt = 'Uploaded Image';
      imgElement.width = 100; // Adjust size as needed
      imgElement.height = 100; // Adjust size as needed
      
      // Ensure the columns are properly populated
      const historyTable = document.getElementById('historyTable');
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${data.id}</td>
        <td>${data.category}</td>
        <td>${data.confidence}</td>
        <td>${data.timestamp}</td>
        <td><img src="${data.image_url}" width="100" height="100" alt="Uploaded Image"></td>
      `;
      historyTable.appendChild(row);

      // Refresh stats and history dynamically
      populateStatistics();
      populateHistory();
    } else {
      alert('Error: ' + data.error);
    }
});

// Populate Classification History
async function populateHistory() {
    const response = await fetch('/history');
    const history = await response.json();

    const historyTable = document.getElementById('historyTable');
    historyTable.innerHTML = history
      .map(
        (entry) => `
        <tr>
          <td>${entry.id}</td>
          <td>${entry.category}</td>
          <td>${entry.confidence}</td>
          <td>${entry.timestamp}</td>
          <td><img src="${entry.image_url}" width="100" height="100" alt="Image"></td>
        </tr>
      `
      )
      .join('');
}

// Populate Statistics and Insights
async function populateStatistics() {
    const response = await fetch('/stats');
    const stats = await response.json();

    const statsContainer = document.getElementById('statistics');
    statsContainer.innerHTML = `
      <p>Total Classifications: ${stats.totalClassifications}</p>
      <p>Recyclable Waste: ${stats.recyclablePercentage.toFixed(2)}%</p>
      <p>Non-Recyclable Waste: ${stats.nonRecyclablePercentage.toFixed(2)}%</p>
    `;
}

// Camera Interface
const cameraStream = document.getElementById('cameraStream');
const snapshotCanvas = document.getElementById('snapshot');
const captureBtn = document.getElementById('captureBtn');
let videoStream;

// Start Camera
navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    videoStream = stream;
    cameraStream.srcObject = stream;
  })
  .catch((error) => {
    console.error('Camera error:', error);
    alert('Unable to access camera.');
  });

captureBtn.addEventListener('click', () => {
    const context = snapshotCanvas.getContext('2d');
    snapshotCanvas.width = cameraStream.videoWidth;
    snapshotCanvas.height = cameraStream.videoHeight;
    context.drawImage(cameraStream, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

    snapshotCanvas.classList.remove('d-none');
    alert('Image captured successfully!');
});

// Initial Page Load
populateStatistics();
populateHistory();
