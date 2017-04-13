import View from 'girder/views/View';

import saveAnnotation from '../templates/dialogs/saveAnnotation.pug';

var SaveAnnotation = View.extend({
    events: {
        'click .h-submit': 'save',
        'submit form': 'save'
    },
    render() {
        this.$el.html(
            saveAnnotation({
                annotation: this.annotation.toJSON()
            })
        ).girderModal(this);
    },
    save(evt) {
        evt.preventDefault();
        this.annotation.set({
            name: this.$('#h-annotation-name').val(),
            description: this.$('#h-annotation-description').val()
        });
        this.annotation.trigger('g:save');
        this.$el.modal('hide');
    }
});

var dialog = new SaveAnnotation({
    parentView: null
});

function show(annotation) {
    dialog.annotation = annotation;
    dialog.setElement('#g-dialog-container').render();
}

export default show;
