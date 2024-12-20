<template>
    <form class="mt-5 mb-3">
        <div class="input-group border border-info rounded">
            <input v-model="formData.url" type="url" class="form-control" placeholder="https://example.com" required>
            <button class="btn btn-main w-25" type="button" :disabled="loading" @click="getReviews">
                <span v-if="loading" class="spinner-border spinner-border-sm text-info" role="status" aria-hidden="true"></span>
                <span v-if="!loading">Get Reviews</span>
            </button>
        </div>
        <div v-if="error" class="alert alert-danger mt-3">{{ error }}</div>
    </form>
</template>

<script>
import axios from 'axios';
export default {
    data() {
        return {
            formData: {
                url: 'https://tiki.vn/giay-da-nam-giay-oxford-cong-so-bui-leather-g103-da-bo-nappa-cao-cap-bao-hanh-12-thang-p134739207.html',
            },
            loading: false,
            error: null
        };
    },
    methods: {
        async getReviews() {
            if (this.loading) return;
            this.loading = true;
            this.error = null;
            try {
                const response = await axios.post('http://127.0.0.1:5000/get_reviews', this.formData, {
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                this.$emit('update-data', response.data);
                // Handle the response as needed
            } catch (error) {
                this.error = 'Failed to fetch reviews. Please try again later.';
                console.error('Error:', error);
            } finally {
                this.loading = false;
            }
        }
    }
}
</script>
<style>
button.btn-main {
    background: #5cfaff;
    color: #000;
}

button.btn-main:hover{
    color: #fff;
}
</style>