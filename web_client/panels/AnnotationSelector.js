import _ from 'underscore';

import { AccessType } from 'girder/constants';
import eventStream from 'girder/utilities/EventStream';
import { getCurrentUser } from 'girder/auth';
import Panel from 'girder_plugins/slicer_cli_web/views/Panel';
import AnnotationModel from 'girder_plugins/large_image/models/AnnotationModel';
import {events as girderEvents} from 'girder';

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
        'click .h-annotation-name': '_editAnnotation',
        'click .h-toggle-annotation': 'toggleAnnotation',
        'click .h-delete-annotation': 'deleteAnnotation',
        'click .h-create-annotation': 'createAnnotation',
        'click .h-edit-annotation-metadata': 'editAnnotationMetadata',
        'click .h-show-all-annotations': 'showAllAnnotations',
        'click .h-hide-all-annotations': 'hideAllAnnotations',
        'mouseenter .h-annotation': '_highlightAnnotation',
        'mouseleave .h-annotation': '_unhighlightAnnotation',
        'change #h-toggle-labels': 'toggleLabels',
        'change #h-toggle-interactive': 'toggleInteractiveMode',
        'input #h-annotation-opacity': '_changeGlobalOpacity',
        'input #h-annotation-fill-opacity': '_changeGlobalFillOpacity',
        'click .h-annotation-group-name': '_toggleExpandGroup'
    }),

    /**
     * Create the panel.
     *
     * @param {object} settings
     * @param {AnnotationCollection} settings.collection
     *     The collection representing the annotations attached
     *     to the current image.
     */
    initialize(settings = {}) {
        this._expandedGroups = new Set();
        this._opacity = settings.opacity || 0.9;
        this._fillOpacity = settings.fillOpacity || 1.0;
        this._showAllAnnotationsState = false;
        this.listenTo(this.collection, 'sync remove update reset change:displayed change:loading', this.render);
        this.listenTo(this.collection, 'change:highlight', this._changeAnnotationHighlight);
        this.listenTo(eventStream, 'g:event.job_status', _.debounce(this._onJobUpdate, 500));
        this.listenTo(eventStream, 'g:eventStream.start', this._refreshAnnotations);
        this.listenTo(this.collection, 'change:annotation change:groups', this._saveAnnotation);
        this.listenTo(girderEvents, 'g:login', () => {
            this.collection.reset();
            this._parentId = undefined;
        });
    },

    render() {
        const annotationGroups = this._getAnnotationGroups();
        this.$('[data-toggle="tooltip"]').tooltip('destroy');
        if (!this.viewer) {
            this.$el.empty();
            return;
        }
        this.$el.html(annotationSelectorWidget({
            id: 'annotation-panel-container',
            title: 'Annotations',
            activeAnnotation: this._activeAnnotation ? this._activeAnnotation.id : '',
            showLabels: this._showLabels,
            user: getCurrentUser() || {},
            writeAccess: this._writeAccess,
            opacity: this._opacity,
            fillOpacity: this._fillOpacity,
            interactiveMode: this._interactiveMode,
            expandedGroups: this._expandedGroups,
            annotationGroups,
            _
        }));
        this.$('.s-panel-content').collapse({toggle: false});
        this.$('[data-toggle="tooltip"]').tooltip({container: 'body'});
        this._changeGlobalOpacity();
        this._changeGlobalFillOpacity();
        if (this._showAllAnnotationsState) {
            this.showAllAnnotations();
        }
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

        // any manual change in the display state will reset the "forced display" behavior
        this._showAllAnnotationsState = false;
        model.set('displayed', !model.get('displayed'));
        if (!model.get('displayed')) {
            model.unset('highlight');
        }
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
                    model.unset('highlight');
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
            () => {
                model.save().done(() => {
                    if (model.get('displayed')) {
                        this.trigger('h:redraw', model);
                    }
                });
            }
        );
    },

    _onJobUpdate(evt) {
        if (this.parentItem && evt.data.status > 2) {
            this._refreshAnnotations();
        }
    },

    _refreshAnnotations() {
        if (!this.parentItem || !this.parentItem.id) {
            return;
        }
        var models = this.collection.indexBy(_.property('id'));
        this.collection.offset = 0;
        this.collection.fetch({itemId: this.parentItem.id}).then(() => {
            var activeId = (this._activeAnnotation || {}).id;
            this.collection.each((model) => {
                if (!_.has(models, model.id)) {
                    model.set('displayed', true);
                } else {
                    if (models[model.id].get('displayed')) {
                        if (model.get('_version') !== models[model.id].get('_version')) {
                            model.refresh(true);
                            model.set('displayed', true);
                        }
                    }
                    // set without triggering a change; a change reloads and
                    // rerenders, which is only done if it has changed (above)
                    model.attributes.displayed = models[model.id].get('displayed');
                }
            });
            this.render();
            this._activeAnnotation = null;
            if (activeId) {
                this._setActiveAnnotation(this.collection.get(activeId));
            }
            // remove annotations that are displayed but have been deleted
            Object.keys(models).forEach((id) => {
                if (!this.collection.get(id) && models[id].get('displayed')) {
                    this.viewer.removeAnnotation(models[id]);
                }
            });
            return null;
        });
    },

    toggleLabels(evt) {
        this._showLabels = !this._showLabels;
        this.trigger('h:toggleLabels', {
            show: this._showLabels
        });
    },

    toggleInteractiveMode(evt) {
        this._interactiveMode = !this._interactiveMode;
        this.trigger('h:toggleInteractiveMode', this._interactiveMode);
    },

    interactiveMode() {
        return this._interactiveMode;
    },

    _editAnnotation(evt) {
        var id = $(evt.currentTarget).parents('.h-annotation').data('id');
        this.editAnnotation(this.collection.get(id));
    },

    editAnnotation(model) {
        // deselect the annotation if it is already selected
        if (this._activeAnnotation && model && this._activeAnnotation.id === model.id) {
            this._activeAnnotation = null;
            this.trigger('h:editAnnotation', null);
            this.render();
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
        this._setActiveAnnotation(model);
    },

    _setActiveAnnotation(model) {
        this._activeAnnotation = model;

        if (!((model.get('annotation') || {}).elements || []).length) {
            // Only load the annotation if it hasn't already been loaded.
            // Technically, an annotation *could* have 0 elements, in which
            // case loading it again should be quick.  There doesn't seem
            // to be any other way to detect an unloaded annotation.
            model.set('loading', true);
            model.fetch().done(() => {
                this._setActiveAnnotationWithoutLoad(model);
            }).always(() => {
                model.unset('loading');
            });
        } else {
            this._setActiveAnnotationWithoutLoad(model);
        }
    },

    _setActiveAnnotationWithoutLoad(model) {
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
        if (!annotation._saving && !annotation._inFetch && !annotation.get('loading')) {
            annotation._saving = true;
            this.trigger('h:redraw', annotation);
            annotation.save().always(() => {
                annotation._saving = false;
            });
        }
    },

    _writeAccess(annotation) {
        return annotation.get('_accessLevel') >= AccessType.ADMIN;
    },

    showAllAnnotations() {
        this._showAllAnnotationsState = true;
        this.collection.each((model) => {
            model.set('displayed', true);
        });
    },

    hideAllAnnotations() {
        this._showAllAnnotationsState = false;
        this.collection.each((model) => {
            model.set('displayed', false);
        });
    },

    _highlightAnnotation(evt) {
        const id = $(evt.currentTarget).data('id');
        const model = this.collection.get(id);
        if (model.get('displayed')) {
            this.parentView.trigger('h:highlightAnnotation', id);
        }
    },

    _unhighlightAnnotation() {
        this.parentView.trigger('h:highlightAnnotation');
    },

    _changeAnnotationHighlight(model) {
        this.$(`.h-annotation[data-id="${model.id}"]`).toggleClass('h-highlight-annotation', model.get('highlighted'));
    },

    _changeGlobalOpacity() {
        this._opacity = this.$('#h-annotation-opacity').val();
        this.$('.h-annotation-opacity-container')
            .attr('title', `Annotation total opacity ${(this._opacity * 100).toFixed()}%`);
        this.trigger('h:annotationOpacity', this._opacity);
    },

    _changeGlobalFillOpacity() {
        this._fillOpacity = this.$('#h-annotation-fill-opacity').val();
        this.$('.h-annotation-fill-opacity-container')
            .attr('title', `Annotation fill opacity ${(this._fillOpacity * 100).toFixed()}%`);
        this.trigger('h:annotationFillOpacity', this._fillOpacity);
    },

    _toggleExpandGroup(evt) {
        const name = $(evt.currentTarget).parent().data('groupName');
        if (this._expandedGroups.has(name)) {
            this._expandedGroups.delete(name);
        } else {
            this._expandedGroups.add(name);
        }
        this.render();
    },

    _getAnnotationGroups() {
        // Annotations without elements don't have any groups, so we inject the null group
        // so that they are displayed in the panel.
        this.collection.each((a) => {
            const groups = a.get('groups') || [];
            if (!groups.length) {
                groups.push(null);
            }
        });
        const groupObject = {};
        const groups = _.union(...this.collection.map((a) => a.get('groups')));
        _.each(groups, (group) => {
            const groupList = this.collection.filter(
                (a) => _.contains(a.get('groups'), group));

            if (group === null) {
                group = 'Other';
            }
            groupObject[group] = _.sortBy(groupList, (a) => a.get('created'));
        });
        this._triggerGroupCountChange(groupObject);
        return groupObject;
    },

    _triggerGroupCountChange(groups) {
        const groupCount = {};
        _.each(groups, (annotations, name) => {
            if (name !== 'Other') {
                groupCount[name] = annotations.length;
            } else {
                groupCount.default = annotations.length;
            }
        });
        this.trigger('h:groupCount', groupCount);
    }
});

export default AnnotationSelector;
