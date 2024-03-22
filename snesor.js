// Dummy sensor data (replace with real data from IoT sensors)
const sensorData = [
    { name: "Sensor 1", value: 50 },
    { name: "Sensor 2", value: 75 },
    { name: "Sensor 3", value: 60 }
  ];
  
  // Display sensor data on the webpage
  const sensorContainer = document.getElementById('sensor-data');
  sensorData.forEach(sensor => {
    const sensorElement = document.createElement('div');
    sensorElement.classList.add('sensor');
    sensorElement.innerHTML = `
      <h3>${sensor.name}</h3>
      <p>Energy Consumption: ${sensor.value} kW/h</p>
    `;
    sensorContainer.appendChild(sensorElement);
  });
// Fetch sensor data from the backend (JSON file)
fetch('sensor_data.json')
  .then(response => response.json())
  .then(data => {
    // Display sensor data on the webpage
    const sensorContainer = document.getElementById('sensor-data');
    data.forEach(sensor => {
      const sensorElement = document.createElement('div');
      sensorElement.classList.add('sensor');
      sensorElement.innerHTML = `
        <h3>${sensor.name}</h3>
        <p>Energy Consumption: ${sensor.value} kW/h</p>
      `;
      sensorContainer.appendChild(sensorElement);
    });
  })
  .catch(error => console.error('Error fetching sensor data:', error));
// Fetch sensor data from the backend (JSON file)
fetch('sensor_data.json')
  .then(response => response.json())
  .then(data => {
    // Display sensor data on the webpage
    const sensorContainer = document.getElementById('sensor-data');
    data.forEach(sensor => {
      const sensorElement = document.createElement('div');
      sensorElement.classList.add('sensor');
      sensorElement.innerHTML = `
        <h3>${sensor.name}</h3>
        <p>Location: ${sensor.location}</p>
        <p>Energy Consumption: ${sensor.value} kW/h</p>
      `;
      sensorContainer.appendChild(sensorElement);
    });

    // Search functionality
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchResult = document.getElementById('search-result');

    searchButton.addEventListener('click', () => {
      const searchTerm = searchInput.value.trim().toLowerCase();
      const foundSensor = data.find(sensor => sensor.name.toLowerCase() === searchTerm);
      if (foundSensor) {
        searchResult.textContent = `Sensor "${foundSensor.name}" found. Location: ${foundSensor.location}. Energy Consumption: ${foundSensor.value} kW/h`;
      } else {
        searchResult.textContent = 'Sensor not found.';
      }
    });
  })
  .catch(error => console.error('Error fetching sensor data:', error));
  