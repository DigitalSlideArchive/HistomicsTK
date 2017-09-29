import ItemModel from 'girder/models/ItemModel';
import View from 'girder/views/View';
import { wrap } from 'girder/utilities/PluginUtils';

import TaskRunView from 'girder_plugins/item_tasks/views/TaskRunView';

import events from '../../events';
import router from '../../router';
import JobsPanel from '../../panels/JobsPanel';

import taskInfoPanel from '../../templates/panels/taskInfoPanel.pug';
import '../../stylesheets/layout/taskPanelGroup.styl';

// Wrap the task run view to reenable the execute button after job submission.
wrap(TaskRunView, 'execute', function (execute, ...args) {
    const returnVal = execute.apply(this, args);
    this.$('.g-run-task').girderEnable(true);
    return returnVal;
});

const TaskPanelGroup = View.extend({
    initialize() {
        this.model = this.model || new ItemModel();
        this.listenTo(this.model, 'g:fetched', this.render);
        this.listenTo(events, 'query:analysis', this.setAnalysis);
    },

    render() {
        this.$el.empty();
        if (this.model.isNew() || !this.model.has('meta')) {
            return;
        }

        if (!this.model.get('meta') || !this.model.get('meta').isItemTask) {
            events.trigger('g:alert', {
                text: 'Unknown or invalid task.',
                type: 'danger',
                timeout: 5000,
                icon: 'attention'
            });
            this.model.clear();
            router.setQuery('analysis', null);
            return;
        }

        this.taskRunView = new TaskRunView({
            parentView: this,
            model: this.model,
            el: this.el
        }).render();

        this.$('.g-panel-group').prepend(taskInfoPanel());
        this.jobsPanel = new JobsPanel({
            parentView: this,
            el: this.$('.h-jobs-panel')
        }).render();

        const title = this.$('.g-body-title').remove();
        const description = this.$('.g-task-description-container').remove();
        const execute = this.$('.g-run-container').remove();

        // mutate the title link to the correct route
        title.find('a')
            .attr('href', `/girder${title.find('a').attr('href')}`)
            .attr('target', '_blank');

        // make the title button smaller
        execute.find('button').removeClass('btn-lg');

        // move the title, description, execution elements to a new panel
        this.$('.h-task-title').append(title);
        this.$('.h-info-panel-description').append(description);
        this.$('.h-info-panel-buttons').append(execute);
        return this;
    },

    setAnalysis(taskId) {
        if (taskId) {
            this.model.set({_id: taskId}).fetch();
        }
    }
});

export default TaskPanelGroup;
