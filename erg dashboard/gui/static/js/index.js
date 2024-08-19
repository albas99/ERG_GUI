let form = document.getElementById('form');

function uploadFile() {
    let formData = new FormData()
    let file = document.getElementsByName('sigfile')[0].files[0];
    formData.append('sigfile', file, file.name);
    let xhr = new XMLHttpRequest();
    xhr.open("POST", '/upload/', true);
    xhr.onload = function () {
        if (xhr.status === 200) {
            let response = JSON.parse(xhr.responseText);
            let file_url = response.file_url;
            console.log(xhr.responseText);
            fetch(file_url).then(
                response => response.text()
            ).then(
                csvData => {
                    const df = Papa.parse(csvData).data;
                    const columnNames = df[0];

                    let select = document.getElementById('signalSelect');
                    columnNames.forEach(name => {
                        let option = document.createElement('option');
                        option.value = name;
                        option.text = name;
                        select.appendChild(option);
                    });

                    select.addEventListener('change', function () {
                        const selectedColumn = select.value;
                        fetch('/process_column', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                column_name: selectedColumn,
                                file_url: file_url
                            })
                        }).then(
                            response => {
                                if (!response.ok) {
                                    throw new Error('Network response was not ok!')
                                }
                                return response.json();
                            }
                        ).then(data => {
                            console.log(data);
                        }).catch(error => {
                            console.error('Error:', error);
                        });
                    });
                });
        } else {
            alert('An error ocurred!');
        }
    };
    xhr.setRequestHeader('X-CSRFToken', getCookie('csrftoken'));
    xhr.send(formData);
}

form.addEventListener('submit', function (e) {
    e.preventDefault();
    console.log('Form submitted')
    uploadFile();
});