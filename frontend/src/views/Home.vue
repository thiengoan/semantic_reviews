<template>
  <div class="container relative">
    <h1 class="text-center bg-title">TOOL SEMANTIC REVIEWS TIKI</h1>
    <Form @update-data="updateData" />
    <Card v-if="data" :data="data"/>
  </div>
</template>

<script>

import Form from "../components/From.vue";
import Card from "../components/Card.vue";

export default {
  components: { Form, Card },
  data() {
    return {
      data: null,
    };
  },
  methods: {
    updateData(data) {
      this.data = data;
    },
  },
};
</script>
<style>
.bg-title{
  background: linear-gradient(90deg, #c5fbff .17%, #89fff8 18.52%, #4dffa7 54.29%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  font-size: 2.5rem;
  font-weight: bold;
}
</style>