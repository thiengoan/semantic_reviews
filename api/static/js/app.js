$('#submitFrm').on('click', function() {
    // Lấy URL từ input
    const url = $('#urlInput').val();
    const model = $('#modelType').val();

    alert(model,url);
    // Gửi URL đến server Flask để lấy kết quả
    $.ajax({
        url: 'http://127.0.0.1:5000/predict',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ url: url, model: model }),
        success: function(data) {
            if (data.predictions) {
                $('#predictions').empty(); // Clear previous results
                // Mock data for the chart
                var positive = 0;
                var negative = 0;
                data.predictions.forEach(function(item) {
                    positive = item.prediction == 1 ? positive + 1 : positive;
                    negative = item.prediction == 0 ? negative + 1 : negative;
                    const predictionElem = $('<div></div>');
                    predictionElem.html(`<strong>Đánh giá:</strong> ${item.review}<br>
                                          <strong>Nhận định:</strong> ${item.prediction == 1 ? 'Tích cực' : 'Tiêu cực'}<br>
                                          <strong>Độ tin cậy:</strong> ${item.confidence.toFixed(2)}<br><hr>`);
                    $('#predictions').append(predictionElem);
                });

                // Hiển thị biểu đồ
                showChart(positive*100, negative*100);

            } else {
                alert('Không có đánh giá nào hoặc lỗi xảy ra.');
            }
        },
        error: function(error) {
            console.error('Error:', error);
            alert('Đã có lỗi xảy ra. Vui lòng thử lại!');
        }
    });
});

function showChart(positive,negative){
    const ctx = document.getElementById('resultChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Positive', 'Negative'],
            datasets: [{
                label: 'Score',
                data: [positive, negative],
                backgroundColor: [
                    '#4caf50', // Positive
                    '#f44336', // Negative
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}
