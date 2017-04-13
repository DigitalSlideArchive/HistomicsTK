import _ from 'underscore';

import events from 'girder/events';

import AnnotationModel from 'girder_plugins/large_image/models/AnnotationModel';
import Panel from 'girder_plugins/slicer_cli_web/views/Panel';
import editAnnotation from '../dialogs/editAnnotation';
import saveAnnotation from '../dialogs/saveAnnotation';

import drawWidget from '../templates/panels/drawWidget.pug';
import '../stylesheets/panels/drawWidget.styl';

var DrawWidget = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'click .h-save-annotation': 'saveAnnotation',
        'click .h-edit-element': 'editElement',
        'click .h-delete-element': 'deleteElement',
        'click .h-draw': 'drawElement'
    }),
    initialize() {
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
        this.$('.s-panel-content').collapse({toggle: false});
        this.$('[data-toggle="tooltip"]').tooltip({container: 'body'});
        return this;
    },
    setViewer(viewer) {
        this.viewer = viewer;
        return this;
    },
    saveAnnotation(evt) {
        saveAnnotation(this.annotation);
    },
    editElement(evt) {
        editAnnotation(this.collection.get(this._getId(evt)));
    },
    deleteElement(evt) {
        this.collection.remove(this._getId(evt));
    },
    drawElement(evt) {
        var $el = this.$(evt.target);
        var type = $el.data('type');
        return this.viewer.startDrawMode(type)
            .then((element) => this._onCreate(element));
    },
    reset() {
        this.collection.reset();
    },
    _onCreate(element) {
        this.collection.add(element);
    },
    _onCollectionChange() {
        this.viewer.drawAnnotation(this.collection.annotation);
        this.render();
    },
    _getId(evt) {
        return this.$(evt.target).parent('.h-element').data('id');
    },
    _onSaveAnnotation() {
        events.trigger('h:createAnnotation', this.annotation);
        this.reset();
    }
});

export default DrawWidget;
