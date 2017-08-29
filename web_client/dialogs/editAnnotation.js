import tinycolor from 'tinycolor2';

import View from 'girder/views/View';

import editAnnotation from '../templates/dialogs/editAnnotation.pug';
import 'girder/utilities/jquery/girderModal';

/**
 * Create a modal dialog with fields to edit the properties of
 * an annotation element.
 */
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
        return this;
    },

    /**
     * Get all data from the form and set the attributes of the
     * attached ElementModel (triggering a change event).
     */
    getData(evt) {
        evt.preventDefault();

        var data = {};
        var label = this.$('#h-element-label').val();
        var validation = '';

        if (label) {
            data.label = {
                value: label
            };
        }

        var lineWidth = this.$('#h-element-line-width').val();
        if (lineWidth) {
            data.lineWidth = parseFloat(lineWidth);
            if (data.lineWidth < 0 || !isFinite(data.lineWidth)) {
                validation += 'Invalid line width. ';
                this.$('#h-element-line-width').parent().addClass('has-error');
            }
        }

        var lineColor = this.$('#h-element-line-color').val();
        if (lineColor) {
            data.lineColor = this.convertColor(lineColor);
        }

        var fillColor = this.$('#h-element-fill-color').val();
        if (fillColor) {
            data.fillColor = this.convertColor(fillColor);
        }

        if (validation) {
            this.$('.g-validation-failed-message').text(validation)
                .removeClass('hidden');
            return;
        }

        this.annotationElement.set(data);
        this.$el.modal('hide');
    },

    /**
     * A helper function converting a string into normalized rgb/rgba
     * color value.  If no value is given, then it returns a color
     * with opacity 0.
     */
    convertColor(val) {
        if (!val) {
            return 'rgba(0,0,0,0)';
        }
        return tinycolor(val).toRgbString();
    }
});

/**
 * Create a singleton instance of this widget that will be rendered
 * when `show` is called.
 */
var dialog = new EditAnnotation({
    parentView: null
});

/**
 * Show the edit dialog box.  Watch for change events on the passed
 * `ElementModel` to respond to user submission of the form.
 *
 * @param {ElementModel} annotationElement The element to edit
 * @returns {EditAnnotation} The dialog's view
 */
function show(annotationElement) {
    dialog.annotationElement = annotationElement;
    dialog.setElement('#g-dialog-container').render();
    return dialog;
}

export default show;
