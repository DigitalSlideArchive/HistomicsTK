import _ from 'underscore';

import eventStream from 'girder/utilities/EventStream';
import Panel from 'girder_plugins/slicer_cli_web/views/Panel';

import annotationSelectorWidget from '../templates/panels/annotationSelector.pug';
import '../stylesheets/panels/annotationSelector.styl';

var AnnotationSelector = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'click .h-annotation > span': 'toggleAnnotation'
    }),
    initialize(settings) {
        this.listenTo(this.collection, 'all', this.render);
        this.listenTo(eventStream, 'g:event.job_status', function (evt) {
            if (this.parentItem && evt.data.status > 2) {
                this.setItem(this.parentItem);
            }
        });
        if (settings.parentItem) {
            this.setItem(settings.parentItem);
        }
    },
    setItem(item) {
        this.parentItem = item;
        if (!this.parentItem || !this.parentItem.id) {
            this.collection.reset();
            this.render();
            return;
        }
        this.collection.offset = 0;
        this.collection.fetch({itemId: item.id})
            .then(() => this.render());
    },
    setViewer(viewer) {
        this.viewer = viewer;
        return this;
    },
    render() {
        if (!this.viewer) {
            this.$el.empty();
            return;
        }
        this.$el.html(annotationSelectorWidget({
            annotations: this.collection.toArray(),
            id: 'annotation-panel-container',
            title: 'Annotations'
        }));
        this.$('.s-panel-content').collapse({toggle: false});
        return this;
    },
    toggleAnnotation(evt) {
        var id = $(evt.currentTarget).data('id');
        var model = this.collection.get(id);
        model.set('displayed', !model.get('displayed'));
        this.render();
    }
});

export default AnnotationSelector;
