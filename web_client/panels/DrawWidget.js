import _ from 'underscore';

import { restRequest } from 'girder/rest';
import AnnotationModel from 'girder_plugins/large_image/models/AnnotationModel';
import Panel from 'girder_plugins/item_tasks/views/Panel';

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
    },

    render() {
        this.$('[data-toggle="tooltip"]').tooltip('destroy');
        if (!this.viewer) {
            this.$el.empty();
            return;
        }
        this.$el.html(drawWidget({
            title: 'Draw',
            elements: this.collection.toJSON()
        }));
        this.$('.g-panel-content').collapse({toggle: false});
        this.$('[data-toggle="tooltip"]').tooltip({container: 'body'});
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
     */
    drawElement(evt) {
        var $el = this.$(evt.currentTarget);
        var type = $el.data('type');
        return this.viewer.startDrawMode(type)
            .then((element) => this.collection.add(element));
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
        console.log('_onSave'); // DWM::
        var data = this.annotation.toJSON();
        data.elements = data.annotation.elements;
        delete data.annotation;
        console.log(['_onSave - ', data, this.image]); // DWM::
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
