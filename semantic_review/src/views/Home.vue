<template>
  <div class="container">
    <h1 class="text-center">Vue Frontend Resource</h1>
    <Form @update-chart="updateChartData" />
    <Chart v-if="chartData" :data="chartData" />
    <div v-if="chartData" class="chart-data">
        <div class="col" v-for="(value, key) in chartData" :key="key">
            <div class="card w-100">{{ value.review }}</div>
        </div>
    </div>
  </div>
</template>

<script>
import Form from "../components/From.vue";
import Chart from "../components/Chart.vue";

export default {
  components: { Form, Chart },
  data() {
    return {
      chartData: null,
    };
  },
  methods: {
    updateChartData(data) {
      this.chartData = data.prediction;
    }
  },
};
</script>
