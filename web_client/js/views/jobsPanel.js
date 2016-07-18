histomicstk.views.JobsPanel = histomicstk.views.Panel.extend({
    events: _.extend(histomicstk.views.Panel.prototype.events, {
        'g:login': 'render',
        'g:login-changed': 'render',
        'g:logout': 'render'
    }),
    initialize: function (settings) {
        this.spec = settings.spec;
        this.listenTo(histomicstk.events, 'h:submit', function () {
            this._jobsListWidget.collection.fetch(undefined, true);
        });
    },
    render: function () {
        var CE = girder.views.jobs_JobListWidget.prototype.columnEnum;
        var columns =  CE.COLUMN_STATUS_ICON | CE.COLUMN_TITLE;

        histomicstk.views.Panel.prototype.render.apply(this, arguments);

        if (girder.currentUser) {
            if (!this._jobsListWidget) {
                this._jobsListWidget = new girder.views.jobs_JobListWidget({
                    columns: columns,
                    showHeader: false,
                    pageLimit: 5,
                    showPaging: false,
                    triggerJobClick: true,
                    parentView: this
                });
                this.listenTo(this._jobsListWidget, 'g:jobClicked', function (job) {
                    // when clicking on a job open to girder's job view in a new window
                    window.open(
                        '/#job/' + job.id,
                        '_blank'
                    );
                });
            }
            this._jobsListWidget.setElement(this.$('.h-panel-content')).render();
        }
    }
});
