<template>
    <div>
        <Bar :data="chartData" :options="chartOptions" />
    </div>
</template>

<script>
import { Bar } from "vue-chartjs";
import { Chart as ChartJS, Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale } from "chart.js";

// Register the necessary components with Chart.js
ChartJS.register(Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale);

export default {
    props: ["data"],
    components: {
        Bar,
    },
    data() {
        return {
            // Dữ liệu cho biểu đồ
            chartData: {
                labels: ["Positive", "Negative"],
                datasets: [
                    {
                        label: "Sentiment",
                        data: [this.data.positive, this.data.negative],
                        backgroundColor: "rgba(75, 192, 192, 0.5)",
                    },
                ],
            },
            // Tùy chỉnh biểu đồ
            chartOptions: {
                responsive: true,
                plugins: {
                    legend: {
                        display: true,
                        position: "top",
                    },
                    title: {
                        display: true,
                        text: "Reviews Sentiment",
                    },
                },
            },
        };
    },
};
</script>