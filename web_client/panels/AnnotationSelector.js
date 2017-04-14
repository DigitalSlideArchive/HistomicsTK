import _ from 'underscore';

import eventStream from 'girder/utilities/EventStream';
import Panel from 'girder_plugins/slicer_cli_web/views/Panel';

import annotationSelectorWidget from '../templates/panels/annotationSelector.pug';
import '../stylesheets/panels/annotationSelector.styl';

/**
 * Create a panel controlling the visibility of annotations
 * on the image view.
 */
var AnnotationSelector = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'click .h-toggle-annotation': 'toggleAnnotation',
        'click .h-delete-annotation': 'deleteAnnotation'
    }),

    /**
     * Create the panel.
     *
     * @param {object} settings
     * @param {AnnotationCollection} settings.collection
     *     The collection representing the annotations attached
     *     to the current image.
     * @param {ItemModel} settings.parentItem
     *     The currently active "large_image" item.
     */
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

    render() {
        this.$('[data-toggle="tooltip"]').tooltip('destroy');
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
        this.$('[data-toggle="tooltip"]').tooltip({container: 'body'});
        return this;
    },

    /**
     * Set the ItemModel associated with the annotation collection.
     * As a side effect, this resets the AnnotationCollection and
     * fetches annotations from the server associated with the
     * item.
     *
     * @param {ItemModel} item
     */
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
        return this;
    },

    /**
     * Set the image "viewer" instance.  This should be a subclass
     * of `large_image/imageViewerWidget` that is capable of rendering
     * annotations.
     */
    setViewer(viewer) {
        this.viewer = viewer;
        return this;
    },

    /**
     * Toggle the renderering of a specific annotation.  Sets the
     * `displayed` attribute of the `AnnotationModel`.
     */
    toggleAnnotation(evt) {
        var id = $(evt.currentTarget).parent('.h-annotation').data('id');
        var model = this.collection.get(id);
        model.set('displayed', !model.get('displayed'));
    },

    /**
     * Delete an annotation from the server.
     */
    deleteAnnotation(evt) {
        var id = $(evt.currentTarget).parent('.h-annotation').data('id');
        var model = this.collection.get(id);
        if (model) {
            model.unset('displayed');
            model.destroy();
        }
    }
});

export default AnnotationSelector;
