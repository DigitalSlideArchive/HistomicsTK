import _ from 'underscore';

import events from 'girder/events';
import Panel from 'girder_plugins/slicer_cli_web/views/Panel';

import StyleCollection from '../collections/StyleCollection';
import StyleModel from '../models/StyleModel';
import editElement from '../dialogs/editElement';
import editStyleGroups from '../dialogs/editStyleGroups';
import drawWidget from '../templates/panels/drawWidget.pug';
import '../stylesheets/panels/drawWidget.styl';

/**
 * Create a panel with controls to draw and edit
 * annotation elements.
 */
var DrawWidget = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'click .h-edit-element': 'editElement',
        'click .h-delete-element': 'deleteElement',
        'click .h-draw': 'drawElement',
        'change .h-style-group': '_setStyleGroup',
        'click .h-configure-style-group': '_styleGroupEditor',
        'mouseenter .h-element': '_highlightElement',
        'mouseleave .h-element': '_unhighlightElement'
    }),

    /**
     * Create the panel.
     *
     * @param {object} settings
     * @param {ItemModel} settings.image
     *     The associate large_image "item"
     */
    initialize(settings) {
        this.image = settings.image;
        this.annotation = settings.annotation;
        this.collection = this.annotation.elements();
        this.viewer = settings.viewer;
        this._drawingType = settings.drawingType || null;

        this._highlighted = {};
        this._groups = new StyleCollection();
        this._style = new StyleModel({id: 'default'});
        this.listenTo(this._groups, 'update', this.render);
        this.listenTo(this.collection, 'add remove reset', this._recalculateGroupAggregation);
        this.listenTo(this.collection, 'change update reset', this.render);
        this._groups.fetch().done(() => {
            // ensure the default style exists
            if (this._groups.has('default')) {
                this._style.set(this._groups.get('default').toJSON());
            } else {
                this._groups.add(this._style.toJSON());
                this._groups.get(this._style.id).save();
            }
        });
        this.on('h:mouseon', (model) => {
            if (model && model.id) {
                this._highlighted[model.id] = true;
                this.$(`.h-element[data-id="${model.id}"]`).addClass('h-highlight-element');
            }
        });
        this.on('h:mouseoff', (model) => {
            if (model && model.id) {
                this._highlighted[model.id] = false;
                this.$(`.h-element[data-id="${model.id}"]`).removeClass('h-highlight-element');
            }
        });
    },

    render() {
        this.$('[data-toggle="tooltip"]').tooltip('destroy');
        if (!this.viewer) {
            this.$el.empty();
            return;
        }
        const name = (this.annotation.get('annotation') || {}).name || 'Untitled';
        this.trigger('h:redraw', this.annotation);
        this.$el.html(drawWidget({
            title: 'Draw',
            elements: this.collection.models,
            groups: this._groups,
            style: this._style.id,
            highlighted: this._highlighted,
            name
        }));
        if (this._drawingType) {
            this.$('button.h-draw[data-type="' + this._drawingType + '"]').addClass('active');
            this.drawElement(undefined, this._drawingType);
        }
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
     * Respond to a click on the "edit" button by rendering
     * the EditAnnotation modal dialog.
     */
    editElement(evt) {
        editElement(this.collection.get(this._getId(evt)));
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
            // always show the active annotation when drawing a new element
            this.annotation.set('displayed', true);

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

    cancelDrawMode() {
        this.drawElement(undefined, null);
        this.viewer.annotationLayer._boundHistomicsTKModeChange = false;
        this.viewer.annotationLayer.geoOff(window.geo.event.annotation.state);
    },

    drawingType() {
        return this._drawingType;
    },

    /**
     * Get the element id from a click event.
     */
    _getId(evt) {
        return this.$(evt.currentTarget).parent('.h-element').data('id');
    },

    _setStyleGroup() {
        this._style.set(
            this._groups.get(this.$('.h-style-group').val()).toJSON()
        );
        if (!this._style.get('group') && this._style.id !== 'default') {
            this._style.set('group', this._style.id);
        }
    },

    _styleGroupEditor() {
        editStyleGroups(this._style, this._groups);
    },

    _highlightElement(evt) {
        const id = $(evt.currentTarget).data('id');
        this.parentView.trigger('h:highlightAnnotation', this.annotation.id, id);
    },

    _unhighlightElement() {
        this.parentView.trigger('h:highlightAnnotation');
    },

    _recalculateGroupAggregation() {
        const groups = _.invoke(
            this.collection.filter((el) => el.get('group')),
            'get', 'group'
        );
        this.annotation.set('groups', groups);
    }
});

export default DrawWidget;
