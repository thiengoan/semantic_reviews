<template>
    <form @submit.prevent="handleSubmit" class="mb-3">
      <div class="mb-3">
        <label>Enter URL:</label>
        <input
          v-model="formData.url"
          type="url"
          class="form-control"
          placeholder="https://example.com"
          required
        />
      </div>
      <div class="mb-3">
        <label>Select Option:</label>
        <select v-model="formData.model" class="form-select">
          <option value="pho_bert">PhoBert</option>
        </select>
      </div>
    <button class="btn btn-primary w-100" type="submit" :disabled="loading">
      <span v-if="loading" class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
      <span v-if="!loading">Submit</span>
    </button>
    </form>
</template>
  
  <script>
    import axios from 'axios'
    export default {
        data() {
            return {
                formData: {
                    url: 'https://tiki.vn/giay-da-nam-giay-oxford-cong-so-bui-leather-g103-da-bo-nappa-cao-cap-bao-hanh-12-thang-p134739207.html',
                    model: 'pho_bert'
                },
                loading: false
            };
        },
        methods: {
            async handleSubmit() {
                if (this.loading) return;
                this.loading = true;
                try {
                    const response = await axios({
                        method: 'POST',
                        url: 'http://127.0.0.1:5000/predict',
                        data: this.formData
                    });
                    this.$emit('update-chart', response.data);
                    // Handle the response as needed
                } catch (error) {
                    console.error('Error:', error);
                    // Handle the error as needed
                } finally {
                    this.loading = false;
                }
            }
        },
    };
</script>
  