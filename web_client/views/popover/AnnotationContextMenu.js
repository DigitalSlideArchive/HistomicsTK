import View from '../View';

import template from '../../templates/popover/annotationContextMenu.pug';
import '../../stylesheets/popover/annotationContextMenu.styl';

const AnnotationContextMenu = View.extend({
    events: {
        'click .h-remove-elements': '_removeElements'
    },
    initialize() {
        this.reset();
    },
    render() {
        this.$el.html(template());
        return this;
    },
    reset() {
        if (!this._hovered) {
            return;
        }
        this.parentView.trigger('h:highlightAnnotation');
        this._hovered = null;
    },
    setHovered(element, annotation) {
        const elementModel = annotation.elements().get(element.id);
        if (annotation._pageElements || !elementModel) {
            // ignore context menu actions on paged annotations
            return;
        }
        this.reset();
        this._hovered = { element, annotation };
        this.parentView.trigger('h:highlightAnnotation', annotation.id, element.id);
    },
    _removeElements(evt) {
        evt.preventDefault();
        evt.stopPropagation();

        const { annotation, element } = this._hovered;
        annotation.elements().remove(element);
        this.reset();
        this.trigger('h:close');
    }
});

export default AnnotationContextMenu;
