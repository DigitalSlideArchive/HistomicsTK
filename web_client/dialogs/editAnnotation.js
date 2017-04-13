import tinycolor from 'tinycolor2';

import View from 'girder/views/View';

import editAnnotation from '../templates/dialogs/editAnnotation.pug';
import 'girder/utilities/jquery/girderModal';

var EditAnnotation = View.extend({
    events: {
        'click .h-submit': 'getData',
        'submit form': 'getData'
    },
    render() {
        this.$el.html(
            editAnnotation({
                element: this.annotationElement.toJSON()
            })
        ).girderModal(this);
        this.$('.h-colorpicker').colorpicker();
    },
    getData(evt) {
        evt.preventDefault();

        var data = {};
        var label = this.$('#h-element-label').val();
        if (label) {
            data.label = {
                value: label
            };
        }

        var lineWidth = this.$('#h-element-line-width').val();
        if (lineWidth) {
            data.lineWidth = parseFloat(lineWidth);
        }

        var lineColor = this.$('#h-element-line-color').val();
        if (lineColor) {
            data.lineColor = this.convertColor(lineColor);
        }

        var fillColor = this.$('#h-element-fill-color').val();
        if (fillColor) {
            data.fillColor = this.convertColor(fillColor);
        }

        this.annotationElement.set(data);
        this.$el.modal('hide');
    },
    convertColor(val) {
        if (!val) {
            return 'rgba(0,0,0,0)';
        }
        return tinycolor(val).toRgbString();
    }
});
var dialog = new EditAnnotation({
    parentView: null
});

function show(annotationElement) {
    dialog.annotationElement = annotationElement;
    dialog.setElement('#g-dialog-container').render();
}

export default show;
