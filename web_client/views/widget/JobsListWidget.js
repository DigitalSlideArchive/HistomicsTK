import View from 'girder/views/View';
import { SORT_DESC } from 'girder/constants';
import eventStream from 'girder/utilities/EventStream';

// register worker status definitions as a side effect
import 'girder_plugins/worker/JobStatus';

import JobCollection from 'girder_plugins/jobs/collections/JobCollection';
import JobStatus from 'girder_plugins/jobs/JobStatus';

import jobListWidget from '../../templates/widget/jobsListWidget.pug';

const JobsListWidget = View.extend({
    initialize() {
        if (!this.collection) {
            this.collection = new JobCollection();

            // We want to display 10 jobs, but we are filtering
            // them on the client, so we fetch extra jobs here.
            // Ideally, we would be able to filter them server side
            // but the /job endpoint doesn't currently have the
            // flexibility to do so.
            this.collection.pageLimit = 50;
            this.collection.sortDir = SORT_DESC;
            this.collection.sortField = 'created';
        }

        this.listenTo(this.collection, 'all', this.render);
        this.listenTo(eventStream, 'g:event.job_status', this.fetchAndRender);
        this.listenTo(eventStream, 'g:event.job_created', this.fetchAndRender);
        this.fetchAndRender();
        window.jobs = this;
    },

    render() {
        const jobs = this.collection.filter((job) => {
            return job.get('title') && !job.get('title').startsWith('Pulling');
        }).slice(0, 9);

        this.$('[data-toggle="tooltip"]').tooltip('destroy');
        this.$el.html(jobListWidget({
            jobs,
            JobStatus
        }));
        this.$('[data-toggle="tooltip"]').tooltip({container: 'body'});
        return this;
    },

    fetchAndRender() {
        this.collection.fetch(null, true)
            .then(() => this.render());
    }
});

export default JobsListWidget;
