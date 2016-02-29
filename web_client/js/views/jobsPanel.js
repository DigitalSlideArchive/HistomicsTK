histomicstk.views.JobsPanel = histomicstk.views.Panel.extend({
    events: _.extend(histomicstk.views.Panel.prototype.events, {
        'g:login': 'render',
        'g:login-changed': 'render',
        'g:logout': 'render'
    }),
    initialize: function (settings) {
        this.spec = settings.spec;
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
            }
            this._jobsListWidget.setElement(this.$('.h-panel-content')).render();
        }
    }
});
