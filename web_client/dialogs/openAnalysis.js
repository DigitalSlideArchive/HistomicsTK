import View from 'girder/views/View';
import PaginateTasksWidget from 'girder_plugins/item_tasks/views/PaginateTasksWidget';

import router from '../router';

import openAnalysis from '../templates/dialogs/openAnalysis.pug';
import 'girder/utilities/jquery/girderModal';

var OpenAnalysis = View.extend({
    events: {
    },

    initialize() {
        this.paginateTasksWidget = new PaginateTasksWidget({
            parentView: this
        });
        this.paginateTasksWidget.collection.pageLimit = 5;
        this.listenTo(this.paginateTasksWidget, 'g:selected', this.selectTask);
    },

    render() {
        this.$el.html(
            openAnalysis()
        ).girderModal(this);

        this.paginateTasksWidget.setElement(this.$('.h-task-list-container')).render();
        return this;
    },

    selectTask({task}) {
        if (task && task.id) {
            router.setQuery('analysis', task.id, {trigger: true});
        }
        $('.modal').girderModal('close');
    }
});

/**
 * Create a singleton instance of this widget that will be rendered
 * when `show` is called.
 */
var dialog = new OpenAnalysis({
    parentView: null
});

function show() {
    dialog.setElement('#g-dialog-container').render();
    return dialog;
}

export default show;
