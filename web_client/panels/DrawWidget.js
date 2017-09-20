import _ from 'underscore';

import { restRequest } from 'girder/rest';
import AnnotationModel from 'girder_plugins/large_image/models/AnnotationModel';
import Panel from 'girder_plugins/slicer_cli_web/views/Panel';

import editAnnotation from '../dialogs/editAnnotation';
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
        'click .h-draw': 'drawElement'
    }),

    /**
     * Create the panel.
     *
     * @param {object} settings
     * @param {ItemModel} settings.image
     *     The associate large_image "item"
     */
    initialize(settings) {
        this.annotations = settings.annotations;
        this.image = settings.image;
        this.annotation = new AnnotationModel();
        this.listenTo(this.annotation, 'g:save', this._onSaveAnnotation);
        this.collection = this.annotation.elements();
        this.listenTo(this.collection, 'add remove change reset', this._onCollectionChange);
        this._drawingType = null;
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
            drawingType: this._drawingType
        }));
        this.$('.s-panel-content').collapse({toggle: false});
        this.$('[data-toggle="tooltip"]').tooltip({container: 'body'});
        if (this.viewer.annotationLayer && !this.viewer.annotationLayer._boundHistomicsTKModeChange) {
            this.viewer.annotationLayer._boundHistomicsTKModeChange = true;
            this.viewer.annotationLayer.geoOn(window.geo.event.annotation.mode, (event) => {
                this.$('button.h-draw').removeClass('active');
                if (event.mode) {
                    this.$('button.h-draw[data-type="' + event.mode + '"]').addClass('active');
                }
                if (event.mode !== this._drawingType) {
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
     * Set the image "viewer" instance.  This should be a subclass
     * of `large_image/imageViewerWidget` that is capable of rendering
     * annotations.
     */
    setViewer(viewer) {
        this.viewer = viewer;
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
                .then((element) => this.collection.add(element));
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
        var data = this.annotation.toJSON();
        data.elements = data.annotation.elements;
        delete data.annotation;
        restRequest({
            url: 'annotation?itemId=' + this.image.id,
            contentType: 'application/json',
            processData: false,
            data: JSON.stringify(data),
            type: 'POST'
        }).then((data) => {
            data.displayed = true;
            this.annotations.add(data);
            this.reset();
            return null;
        });
    }
});

export default DrawWidget;
