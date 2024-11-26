<template>
    <div>
      <BarChart :chart-data="chartData" :options="chartOptions" />
    </div>
  </template>
  
  <script>
  import { Bar } from "vue-chartjs";
  import { Chart as ChartJS, Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale } from "chart.js";
  
  ChartJS.register(Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale);
  
  export default {
    components: { BarChart: Bar },
    props: ["data"],
    computed: {
      chartData() {
        return {
          labels: ["Positive", "Negative", "Confidence"],
          datasets: [
            {
              label: "Analysis Results",
              data: this.data,
              backgroundColor: ["#4caf50", "#f44336", "#2196f3"],
            },
          ],
        };
      },
      chartOptions() {
        return {
          responsive: true,
          plugins: {
            legend: { display: false },
          },
          scales: {
            y: { beginAtZero: true },
          },
        };
      },
    },
  };
  </script>
  