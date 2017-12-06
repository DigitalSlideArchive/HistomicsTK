import _ from 'underscore';

import eventStream from 'girder/utilities/EventStream';
import { getCurrentUser } from 'girder/auth';
import Panel from 'girder_plugins/slicer_cli_web/views/Panel';
import AnnotationModel from 'girder_plugins/large_image/models/AnnotationModel';

import events from '../events';
import showSaveAnnotationDialog from '../dialogs/saveAnnotation';

import annotationSelectorWidget from '../templates/panels/annotationSelector.pug';
import '../stylesheets/panels/annotationSelector.styl';

// Too many elements in the draw panel will crash the browser,
// so we only allow editing of annnotations with less than this
// many elements.
const MAX_ELEMENTS_LIST_LENGTH = 5000;

/**
 * Create a panel controlling the visibility of annotations
 * on the image view.
 */
var AnnotationSelector = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'click .h-annotation-name': 'editAnnotation',
        'click .h-toggle-annotation': 'toggleAnnotation',
        'click .h-delete-annotation': 'deleteAnnotation',
        'click .h-create-annotation': 'createAnnotation',
        'click .h-edit-annotation-metadata': 'editAnnotationMetadata',
        'click .h-show-all-annotations': 'showAllAnnotations',
        'click .h-hide-all-annotations': 'hideAllAnnotations',
        'change #h-toggle-labels': 'toggleLabels'
    }),

    /**
     * Create the panel.
     *
     * @param {object} settings
     * @param {AnnotationCollection} settings.collection
     *     The collection representing the annotations attached
     *     to the current image.
     */
    initialize(settings) {
        this.listenTo(this.collection, 'all', this.render);
        this.listenTo(eventStream, 'g:event.job_status', _.debounce(this._onJobUpdate, 500));
        this.listenTo(eventStream, 'g:eventStream.start', this._refreshAnnotations);
        this.listenTo(this.collection, 'change:annotation', this._saveAnnotation);
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
            activeAnnotation: this._activeAnnotation ? this._activeAnnotation.id : '',
            showLabels: this._showLabels,
            user: getCurrentUser() || {},
            writeAccess: this._writeAccess
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
        const id = $(evt.currentTarget).parents('.h-annotation').data('id');
        const model = this.collection.get(id);

        if (model) {
            const name = (model.get('annotation') || {}).name || 'unnamed annotation';
            events.trigger('h:confirmDialog', {
                title: 'Warning',
                message: `Are you sure you want to delete ${name}?`,
                submitButton: 'Delete',
                onSubmit: () => {
                    this.trigger('h:deleteAnnotation', model);
                    model.unset('displayed');
                    this.collection.remove(model);
                    model.destroy();
                }
            });
        }
    },

    editAnnotationMetadata(evt) {
        const id = $(evt.currentTarget).parents('.h-annotation').data('id');
        const model = this.collection.get(id);
        this.listenToOnce(
            showSaveAnnotationDialog(model, {title: 'Edit annotation'}),
            'g:submit',
            () => model.save()
        );
    },

    _onJobUpdate(evt) {
        if (this.parentItem && evt.data.status > 2) {
            this._refreshAnnotations();
        }
    },

    _refreshAnnotations() {
        if (!this.parentItem) {
            return;
        }
        var models = this.collection.indexBy(_.property('id'));
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
    },

    toggleLabels(evt) {
        this._showLabels = !this._showLabels;
        this.trigger('h:toggleLabels', {
            show: this._showLabels
        });
    },

    editAnnotation(evt) {
        var id = $(evt.currentTarget).parents('.h-annotation').data('id');
        var model = this.collection.get(id);
        if (this._activeAnnotation && model && this._activeAnnotation.id === model.id) {
            model.set('displayed', true);
            return;
        }
        if (!this._writeAccess(model)) {
            events.trigger('g:alert', {
                text: 'You do not have write access to this annotation.',
                type: 'warning',
                timeout: 2500,
                icon: 'info'
            });
            return;
        }
        this._activeAnnotation = model;
        model.set('loading', true);
        model.fetch().done(() => {
            const numElements = ((model.get('annotation') || {}).elements || []).length;
            if (this._activeAnnotation && this._activeAnnotation.id !== model.id) {
                return;
            }
            model.set('displayed', true);

            if (numElements > MAX_ELEMENTS_LIST_LENGTH) {
                events.trigger('g:alert', {
                    text: 'This annotation has too many elements to be edited.',
                    type: 'warning',
                    timeout: 5000,
                    icon: 'info'
                });
                this._activeAnnotation = null;
                this.trigger('h:editAnnotation', null);
            } else {
                this.trigger('h:editAnnotation', model);
            }
        }).always(() => {
            model.unset('loading');
        });
    },

    createAnnotation(evt) {
        var model = new AnnotationModel({
            itemId: this.parentItem.id,
            annotation: {}
        });
        this.listenToOnce(
            showSaveAnnotationDialog(model, {title: 'Create annotation'}),
            'g:submit',
            () => {
                model.save().done(() => {
                    model.set('displayed', true);
                    this.collection.add(model);
                    this.trigger('h:editAnnotation', model);
                    this._activeAnnotation = model;
                });
            }
        );
    },

    _saveAnnotation(annotation) {
        if (!this._saving && annotation === this._activeAnnotation) {
            this._saving = true;
            annotation.save().always(() => {
                this._saving = false;
            });
        }
    },

    _writeAccess(annotation) {
        const user = getCurrentUser();
        if (!user || !annotation) {
            return false;
        }
        const admin = user.get && user.get('admin');
        const creator = user.id === annotation.get('creatorId');
        return admin || creator;
    },

    showAllAnnotations() {
        this.collection.each((model) => {
            model.set('displayed', true);
        });
    },

    hideAllAnnotations() {
        this.collection.each((model) => {
            model.set('displayed', false);
        });
    }
});

export default AnnotationSelector;
