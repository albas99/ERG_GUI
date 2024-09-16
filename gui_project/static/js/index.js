
const signalSelectButton = document.getElementById("signalSelectButton")
const signalSelect = document.getElementById("signalSelect")
const url = signalSelect.dataset.url

async function getSelection() {
  let selectedValue = document.getElementById("signalSelect").value
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({selected_value: selectedValue})
    })
    const data = await response.json()
    console.log(data.message)
  } catch (error) {
    console.error('Error:', error)
  }
}

signalSelectButton.addEventListener("click", getSelection)

// signalSelectButton.addEventListener("click", function() {
//   console.log("clicked")
//   const selectedValue = document.getElementById("signalSelect").value
//   console.log(selectedValue)
  
//   fetch("{% url 'get_selection' %}", {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//     },
//     body: JSON.stringify({selected_value: selectedValue})
//     })
//     .then(response => {
//       if (!response.ok) {
//         throw new Error('Network response was not ok')
//       }
//       return response.json()
//     })
//     .then(data => {
//       console.log(data)
//       document.getElementById('response-message').innerText = data.message
//   })
//     .catch(error => {console.error('Error:', error)})
// })