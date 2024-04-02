document.getElementById('uploadForm').addEventListener('submit', function (e) {
    e.preventDefault();
    const fileInput = document.getElementById('csvFile');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    console.log(formData)

    fetch('/upload-csv/', {
        method: 'POST',
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            console.log('Success:', data);
            alert(data.info);
            // Update download link
            const downloadLink = document.getElementById('downloadLink');
            downloadLink.href = `/static/${data.filename}`;
            downloadLink.style.display = 'inline';
        })
        .catch((error) => {
            console.error('Error:', error);
        });
});

function sendChat() {
    const input = document.getElementById('chatInput').value;
    fetch('/csv-agent/invoke', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: input }),
    })
        .then((response) => response.json())
        .then((data) => {
            console.log('Chat response:', data);
            document.getElementById('chatResponse').innerText = data.output;
        })
        .catch((error) => {
            console.error('Chat error:', error);
        });
}
