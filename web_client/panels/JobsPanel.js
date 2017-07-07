import _ from 'underscore';

import { getCurrentUser } from 'girder/auth';
import JobListWidget from 'girder_plugins/jobs/views/JobListWidget';

import events from '../events';
import Panel from './Panel';

var JobsPanel = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'g:login': 'render',
        'g:login-changed': 'render',
        'g:logout': 'render'
    }),
    initialize: function (settings) {
        this.spec = settings.spec;
        this.listenTo(events, 'h:submit', function () {
            this._jobsListWidget.collection.fetch(undefined, true);
        });
    },
    render: function () {
        var CE = JobListWidget.prototype.columnEnum;
        var columns =  CE.COLUMN_STATUS_ICON | CE.COLUMN_TITLE;

        Panel.prototype.render.apply(this, arguments);

        if (getCurrentUser()) {
            if (!this._jobsListWidget) {
                this._jobsListWidget = new JobListWidget({
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
            this._jobsListWidget.setElement(this.$('.s-panel-content')).render();
        }
    }
});

export default JobsPanel;
