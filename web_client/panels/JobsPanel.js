import _ from 'underscore';

import Panel from 'girder_plugins/item_tasks/views/Panel';

import events from '../events';
import JobsListWidget from '../views/widget/JobsListWidget';

var JobsPanel = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'g:login': 'render',
        'g:login-changed': 'render',
        'g:logout': 'render'
    }),
    initialize: function (settings) {
        this.spec = settings.spec;
        this._jobsListWidget = new JobsListWidget({
            parentView: this
        });
        this.listenTo(events, 'h:submit', function () {
            this._jobsListWidget.collection.fetch(undefined, true);
        });
    },
    render: function () {
        Panel.prototype.render.apply(this, arguments);
        this._jobsListWidget.setElement(this.$('.g-panel-content')).render();
        this.$('.g-panel-title').text('Jobs');
    }
});

export default JobsPanel;
