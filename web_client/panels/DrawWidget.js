import _ from 'underscore';

import { getCurrentUser } from 'girder/auth';
import events from 'girder/events';
import AnnotationModel from 'girder_plugins/large_image/models/AnnotationModel';
import Panel from 'girder_plugins/slicer_cli_web/views/Panel';

import StyleCollection from '../collections/StyleCollection';
import StyleModel from '../models/StyleModel';
import editAnnotation from '../dialogs/editAnnotation';
import editStyleGroups from '../dialogs/editStyleGroups';
import saveAnnotation from '../dialogs/saveAnnotation';
import drawWidget from '../templates/panels/drawWidget.pug';
import '../stylesheets/panels/drawWidget.styl';

/**
 * Create a panel with controls to draw and edit
 * annotation elements.
 */
var DrawWidget = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'click .h-save-annotation': 'saveAnnotation',
        'click .h-edit-element': 'editElement',
        'click .h-delete-element': 'deleteElement',
        'click .h-draw': 'drawElement',
        'change .h-style-group': '_setStyleGroup',
        'click .h-configure-style-group': '_styleGroupEditor'
    }),

    /**
     * Create the panel.
     *
     * @param {object} settings
     * @param {ItemModel} settings.image
     *     The associate large_image "item"
     */
    initialize(settings) {
        // autosave at most once every second
        if (DrawWidget.throttleAutosave) {
            this._autoSaveAnnotation = _.throttle(this._autoSaveAnnotation, 1000);
        }

        this.annotations = settings.annotations;
        this.image = settings.image;
        this.annotation = new AnnotationModel({itemId: this.image.id});
        this.collection = this.annotation.elements();
        this.listenTo(this.annotation, 'g:save', this._onSaveAnnotation);
        this.listenTo(this.annotation, 'g:delete', this._onDeleteAnnotation);
        this.listenTo(this.collection, 'change update', this._onCollectionChange);
        this._drawingType = null;

        this._groups = new StyleCollection();
        this._style = new StyleModel({id: 'default'});
        this.listenTo(this._groups, 'update', this.render);
        this._groups.fetch().done(() => {
            // ensure the default style exists
            if (this._groups.has('default')) {
                this._style.set(this._groups.get('default').toJSON());
            } else {
                this._groups.add(this._style.toJSON());
                this._groups.get(this._style.id).save();
            }
        });
        this._savePromise = $.Deferred().resolve().promise();
    },

    render() {
        this.$('[data-toggle="tooltip"]').tooltip('destroy');
        if (!this.viewer) {
            this.$el.empty();
            return;
        }
        this.$el.html(drawWidget({
            title: 'Draw',
            elements: this.collection.toJSON(),
            drawingType: this._drawingType,
            groups: this._groups,
            style: this._style.id
        }));
        this.$('.s-panel-content').collapse({toggle: false});
        this.$('[data-toggle="tooltip"]').tooltip({container: 'body'});
        if (this.viewer.annotationLayer && !this.viewer.annotationLayer._boundHistomicsTKModeChange) {
            this.viewer.annotationLayer._boundHistomicsTKModeChange = true;
            this.viewer.annotationLayer.geoOn(window.geo.event.annotation.mode, (event) => {
                this.$('button.h-draw').removeClass('active');
                if (this._drawingType) {
                    this.$('button.h-draw[data-type="' + this._drawingType + '"]').addClass('active');
                }
                if (event.mode !== this._drawingType && this._drawingType) {
                    /* This makes the draw modes stay on until toggled off.
                     * To turn off drawing after each annotation, add
                     *  this._drawingType = null;
                     */
                    this.drawElement(undefined, this._drawingType);
                }
            });
        }
        return this;
    },

    /**
     * When a region should be drawn that isn't caused by a drawing button,
     * toggle off the drawing mode.
     *
     * @param {event} Girder event that triggered drawing a region.
     */
    _widgetDrawRegion(evt) {
        this._drawingType = null;
        this.$('button.h-draw').removeClass('active');
    },

    /**
     * Set the image "viewer" instance.  This should be a subclass
     * of `large_image/imageViewerWidget` that is capable of rendering
     * annotations.
     */
    setViewer(viewer) {
        this.viewer = viewer;
        // make sure our listeners are in the correct order.
        this.stopListening(events, 's:widgetDrawRegion', this._widgetDrawRegion);
        if (viewer) {
            this.listenTo(events, 's:widgetDrawRegion', this._widgetDrawRegion);
            viewer.stopListening(events, 's:widgetDrawRegion', viewer.drawRegion);
            viewer.listenTo(events, 's:widgetDrawRegion', viewer.drawRegion);
        }
        return this;
    },

    /**
     * Respond to a click on the "Save" button by rendering
     * the SaveAnnotation modal dialog.
     */
    saveAnnotation(evt) {
        saveAnnotation(this.annotation);
    },

    /**
     * Respond to a click on the "edit" button by rendering
     * the EditAnnotation modal dialog.
     */
    editElement(evt) {
        editAnnotation(this.collection.get(this._getId(evt)));
    },

    /**
     * Respond to a click on the "delete" button by removing
     * the element from the element collection.
     */
    deleteElement(evt) {
        this.collection.remove(this._getId(evt));
    },

    /**
     * Respond to clicking an element type by putting the image
     * viewer into "draw" mode.
     *
     * @param {jQuery.Event} [evt] The button click that triggered this event.
     *      `undefined` to use a passed-in type.
     * @param {string|null} [type] If `evt` is `undefined`, switch to this draw
     *      mode.
     */
    drawElement(evt, type) {
        var $el;
        if (evt) {
            $el = this.$(evt.currentTarget);
            $el.tooltip('hide');
            type = $el.hasClass('active') ? null : $el.data('type');
        } else {
            $el = this.$('button.h-draw[data-type="' + type + '"]');
        }
        if (this.viewer.annotationLayer.mode() === type && this._drawingType === type) {
            return;
        }
        if (this.viewer.annotationLayer.mode()) {
            this._drawingType = null;
            this.viewer.annotationLayer.mode(null);
            this.viewer.annotationLayer.geoOff(window.geo.event.annotation.state);
            this.viewer.annotationLayer.removeAllAnnotations();
        }
        if (type) {
            this._drawingType = type;
            this.viewer.startDrawMode(type)
                .then((element) => {
                    this.collection.add(
                        _.map(element, (el) => _.extend(el, _.omit(this._style.toJSON(), 'id')))
                    );
                    return undefined;
                });
        }
    },

    /**
     * Empty all element from the `ElementCollection` and restore the current annotation
     * to its default state.
     */
    reset() {
        this.collection.reset();
        this.annotation.clear();
    },

    /**
     * Respond to changes in the annotation collection by updating
     * the image viewer and rerendering the panel.
     */
    _onCollectionChange() {
        const xhr = this._autoSaveAnnotation();
        if (this.collection.annotation.isNew()) {
            // If the annotation is new, we have to defer drawing it until
            // it is saved to get a valid id for the drawing code to keep
            // track of.
            xhr.done(() => {
                this._drawAnnotation();
            });
        } else {
            this._drawAnnotation();
        }
    },

    /**
     * Get the element id from a click event.
     */
    _getId(evt) {
        return this.$(evt.currentTarget).parent('.h-element').data('id');
    },

    /*
     * Return the active annotation ID.  Other widgets can query this to treat
     * this annotation in a special manner.
     *
     * @return {string} The annotation id, or falsy if no current id.
     */
    getActiveAnnotation() {
        return this.annotation.id;
    },

    /**
     * Respond to a click on the "Save" button by POSTing the
     * annotation to the server and resetting the panel.
     */
    _onSaveAnnotation() {
        this._autoSaveAnnotation().done((data) => {
            data.displayed = true;
            this.annotation.unset('_id');
            this.annotations.add(data);
            this.reset();
            this.render();
        });
    },

    /**
     * Generate a default name for an annotation if none was set by the user.
     * This will be constructed from the user name and a date time string.
     */
    _getDefaultAnnotationName() {
        const date = new Date();
        const user = getCurrentUser().get('login');
        return `${user} ${date.toLocaleString()}`;
    },

    _setStyleGroup() {
        this._style.set(
            this._groups.get(this.$('.h-style-group').val()).toJSON()
        );
    },

    _styleGroupEditor() {
        editStyleGroups(this._style, this._groups);
    },

    /**
     * This method gets called on all changes to the attached collection.
     * The rest calls are prevented from occuring in parallel by using
     * an internal promise.  In addition when not in a testing environment
     * this function is throttled to occur at most once every second to
     * prevent overloading the server.
     */
    _autoSaveAnnotation() {
        this.annotation.set('itemId', this.image.id);
        this._savePromise = this._savePromise.then(() => {
            let xhr;
            const annotationData = this.annotation.get('annotation') || {};

            if (!annotationData.name) {
                annotationData.name = this._getDefaultAnnotationName();
            }

            if (this.collection.isEmpty()) {
                xhr = this.annotation.delete();
            } else {
                xhr = this.annotation.save();
            }

            if (xhr) {
                return xhr.fail((resp) => {
                    // if we fail for any reason, create a new save promise
                    // so we can try again
                    this._savePromise = $.Deferred().resolve().promise();
                    // if the active annotation was deleted by another window,
                    // mark that it is gone so we can create a new one, and
                    // recall the auto save function.
                    if (((resp.responseJSON || {}).message || '').indexOf('Invalid annotation id') === 0) {
                        this.annotation.unset('_id');
                        this._autoSaveAnnotation();
                    }
                });
            }
            return null;
        });
        return this._savePromise;
    },

    _onDeleteAnnotation() {
        this.viewer.removeAnnotation(this.collection.annotation);
    },

    /**
     * Redraw the current annotation as soon as possible.  Due to race conditions
     * with auto and manual saving, it is possible to have a situation where
     * this method is called on a new (no `_id` attribute) annotation despite
     * the effort to prevent that in `_onCollectionChange`.  This retries the
     * redraw in a loop until the next time the model save returns.  In addition,
     * this function prevents unnecessary redrawing by setting a flag while
     * another redraw is queued.
     */
    _drawAnnotation() {
        if (this._drawing) {
            return;
        }

        this._drawing = true;
        if (!this.annotation.isNew()) {
            this.viewer.drawAnnotation(this.annotation);
            this.render();
            this._drawing = false;
        } else {
            window.setTimeout(() => {
                this._drawing = false;
                this._drawAnnotation();
            }, 100);
        }
    }
});

DrawWidget.throttleAutosave = true;

export default DrawWidget;
