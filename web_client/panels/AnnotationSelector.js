import _ from 'underscore';

import eventStream from 'girder/utilities/EventStream';
import Panel from 'girder_plugins/item_tasks/views/Panel';

import annotationSelectorWidget from '../templates/panels/annotationSelector.pug';
import '../stylesheets/panels/annotationSelector.styl';

/**
 * Create a panel controlling the visibility of annotations
 * on the image view.
 */
var AnnotationSelector = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'click .h-toggle-annotation': 'toggleAnnotation',
        'click .h-delete-annotation': 'deleteAnnotation',
        'change #h-toggle-labels': 'toggleLabels'
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
        this.listenTo(eventStream, 'g:event.job_status', _.debounce(this._onJobUpdate, 500));
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
            annotations: this.collection.sortBy('created'),
            id: 'annotation-panel-container',
            title: 'Annotations',
            showLabels: this._showLabels
        }));
        this.$('.g-panel-content').collapse({toggle: false});
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
        if (this._parentId === item.id) {
            return;
        }

        this.parentItem = item;
        this._parentId = item.id;

        if (!this._parentId) {
            this.collection.reset();
            this.render();
            return;
        }
        this.collection.offset = 0;
        this.collection.reset();
        this.collection.fetch({itemId: this._parentId});

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
        var id = $(evt.currentTarget).parents('.h-annotation').data('id');
        var model = this.collection.get(id);
        model.set('displayed', !model.get('displayed'));
    },

    /**
     * Delete an annotation from the server.
     */
    deleteAnnotation(evt) {
        var id = $(evt.currentTarget).parents('.h-annotation').data('id');
        var model = this.collection.get(id);
        if (model) {
            model.unset('displayed');
            this.collection.remove(model);
            model.destroy();
        }
    },

    _onJobUpdate(evt) {
        var models = this.collection.indexBy(_.property('id'));
        if (this.parentItem && evt.data.status > 2) {
            this.collection.offset = 0;
            this.collection.fetch({itemId: this.parentItem.id}).then(() => {
                this.collection.each((model) => {
                    if (!_.has(models, model.id)) {
                        model.set('displayed', true);
                    } else {
                        model.set('displayed', models[model.id].get('displayed'));
                    }
                });
                this.render();
                return null;
            });
        }
    },

    toggleLabels(evt) {
        this._showLabels = !this._showLabels;
        this.trigger('h:toggleLabels', {
            show: this._showLabels
        });
    }
});

export default AnnotationSelector;
