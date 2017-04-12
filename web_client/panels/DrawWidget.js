import _ from 'underscore';

import AnnotationModel from 'girder_plugins/large_image/models/AnnotationModel';
import Panel from 'girder_plugins/slicer_cli_web/views/Panel';

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
        this.annotation = new AnnotationModel({
            '_id': 'draw'
        });
        this.collection = this.annotation.elements();
        this.listenTo(this.collection, 'add remove', this._onCollectionChange);
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
        console.log(evt);
    },
    editElement(evt) {
        console.log(evt);
    },
    deleteElement(evt) {
        var id = this.$(evt.target).parent('.h-element').data('id');
        this.collection.remove(id);
    },
    drawElement(evt) {
        var $el = this.$(evt.target);
        var type = $el.data('type');
        return this.viewer.startDrawMode(type).then((element) => this._onCreate(element));
    },
    _onCreate(element) {
        this.collection.add(element);
    },
    _onCollectionChange() {
        this.viewer.drawAnnotation(this.collection.annotation);
        this.render();
    }
});

export default DrawWidget;
