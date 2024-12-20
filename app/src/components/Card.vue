<template>
    <div class="col">
        <label class="text-white fs-4 mt-3">Reviews</label>
        <textarea v-if="getData" class="form-control border-none" name="reviews" v-model="formData.reviews" rows="10"></textarea>
        <div class="mt-3 d-flex justify-content-between align-items-center">
            <div class="input-group">
                <select class="form-select" v-model="formData.model">
                    <option value="lstm">LSTM</option>
                    <option value="naive_bayes">Naive Bayes</option>
                    <option value="pho_bert">PhoBERT</option>
                    <option value="gpt">GPT4o</option>
                    <option value="knn">KNN</option>
                </select>
                <button class="btn btn-danger w-25" type="button" :disabled="loading" @click="predictData">
                    <span v-if="loading" class="spinner-border spinner-border-sm text-light" role="status" aria-hidden="true"></span>
                    <span v-if="!loading">Data Analyze</span>
                </button>
            </div>
        </div>
        <div v-if="show">
            <label class="text-white fs-4 mt-5">Results</label>
            <div class="bg-white rounded p-3 mb-3">
                <Chart :data="countPositiveNegative" />   
            </div>
            <div v-if="countPositiveNegative.positive > countPositiveNegative.negative" class="alert alert-success text-center">
                <h1>Positive</h1>
            </div>
            <div v-else class="alert alert-danger text-center">
                <h1>Negative</h1>
            </div>
        </div>
        <div v-if="show">
            <label class="text-white fs-4 mt-5">Details</label>
            <div class="bg-white rounded p-3 mb-3">
                <table class="table table-bordered table-striped">
                    <thead class="table-dark">
                        <tr class="text-center">
                            <th scope="col">No.</th>
                            <th scope="col">Review</th>
                            <th scope="col">Prediction</th>
                            <th scope="col">Confidence</th>
                            <th scope="col">Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="(item, index) in resData" :key="index">
                            <td>{{ index + 1 }}</td>
                            <td>{{ item.review }}</td>
                            <td>
                                <select v-model="item.prediction" class="form-select">
                                    <option value="Positive">Positive</option>
                                    <option value="Negative">Negative</option>
                                </select>
                            </td>
                            <td>{{ (item.confidence * 100).toFixed(2) }}%</td>
                            <td>
                                <button class="btn btn-primary btn-sm" @click="addDataset(item)">Add</button>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</template>
<script>
import axios from 'axios'
import Chart from './Chart.vue'
export default {
    props: ["data"],
    components: {
        Chart,
    },
    data() {
        return {
            formData: {
                reviews: this.data.data,
                model: 'pho_bert'
            },
            resData: [],
            loading: false,
            show: false,
            error: null
        }; 
    },
    methods: {
        async predictData() {
            if (this.loading) return;
            this.loading = true;
            this.show = false;
            this.error = null;
            try {
                const response = await axios.post('http://127.0.0.1:5000/predict', this.formData);
                this.resData = response.data.prediction;
                this.show = true;
            } catch (error) {
                this.error = 'Failed to fetch prediction. Please try again later.';
                console.error('Error:', error);
            } finally {
                this.loading = false;
            }
        },
        async addDataset(item) {
            try {
                const response = await axios.post('http://127.0.0.1:5000/add-dataset', {
                    review: item.review,
                    prediction: item.prediction
                });
                if (response.data.status === 'success') {
                    alert('Dataset added successfully.');
                } else {
                    alert('Failed to add dataset.');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while adding datasets.');
            }
        }
    },
    computed: {
        getData() {
            console.log(this.data);
            return this.data;
        },
        countPositiveNegative() {
            const counts = { positive: 0, negative: 0 };
            this.resData.forEach(item => {
                if (item.prediction == "Positive" || item.prediction > 0) {
                    counts.positive++;
                } else {
                    counts.negative++;
                }
            });
            return counts;
        }
    },
};
</script>

<style scoped>
.container {
    max-width: 800px;
    margin: auto;
}
.card {
    border-radius: 10px;
}
textarea {
    border: none;
    resize: none;
}
</style>