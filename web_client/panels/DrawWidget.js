import _ from 'underscore';

import { getCurrentUser } from 'girder/auth';
import { restRequest } from 'girder/rest';
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
        this.annotation = new AnnotationModel();
        this.listenTo(this.annotation, 'g:save', this._onSaveAnnotation);
        this.collection = this.annotation.elements();
        this.listenTo(this.collection, 'add remove change reset', this._onCollectionChange);
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
     * Empty all element from the `ElementCollection`.
     */
    reset() {
        this.collection.reset();
    },

    /**
     * Respond to changes in the annotation collection by updating
     * the image viewer and rerendering the panel.
     */
    _onCollectionChange() {
        this._autoSaveAnnotation();
        this.viewer.drawAnnotation(this.collection.annotation);
        this.render();
    },

    /**
     * Get the element id from a click event.
     */
    _getId(evt) {
        return this.$(evt.currentTarget).parent('.h-element').data('id');
    },

    /**
     * Respond to a click on the "Save" button by POSTing the
     * annotation to the server and resetting the panel.
     */
    _onSaveAnnotation() {
        this._autoSaveAnnotation().done((data) => {
            this._activeAnnotationId = null;
            data.displayed = true;
            this.annotations.add(data);
            this.reset();
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

    /**
     * Get a JSON serializaton of the element collection attached to this
     * view.
     */
    _getAnnotationData() {
        var data = _.defaults(
            this.annotation.toJSON(),
            {name: this._getDefaultAnnotationName()}
        );

        // Process elements to remove empty labels which don't validate according
        // to the annotation schema.
        data.elements = _.map(data.annotation.elements, (element) => {
            if (element.label && !element.label.value) {
                delete element.label;
            }
            return element;
        });
        delete data.annotation;
        return data;
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
        this._savePromise = this._savePromise.then(() => {
            const data = this._getAnnotationData();
            let url;
            let type;

            // On the first call to this method `this._activeAnnotationId` will
            // be unset and a new annotation will be generated.  Otherwise,
            // the old annotation will be updated.  In addition, if all
            // elements were deleted, then the autosaved annotation will instead
            // be deleted from the server.
            if (this._activeAnnotationId) {
                url = `annotation/${this._activeAnnotationId}`;
                type = data.elements.length ? 'PUT' : 'DELETE';
            } else if (data.elements.length) {
                url = `annotation?itemId=${this.image.id}`;
                type = 'POST';
            }

            if (url) {
                return restRequest({
                    contentType: 'application/json',
                    processData: false,
                    data: JSON.stringify(data),
                    url,
                    type
                }).done((annotation) => {
                    // Set or delete the current annotation id on return.
                    if (annotation) {
                        this._activeAnnotationId = annotation._id;
                    } else {
                        delete this._activeAnnotationId;
                    }
                });
            }
            return null;
        });
        return this._savePromise;
    }
});

DrawWidget.throttleAutosave = true;

export default DrawWidget;
