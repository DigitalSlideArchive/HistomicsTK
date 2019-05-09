import StyleCollection from '../../collections/StyleCollection';
import View from '../View';

import template from '../../templates/popover/annotationContextMenu.pug';
import '../../stylesheets/popover/annotationContextMenu.styl';

const AnnotationContextMenu = View.extend({
    events: {
        'click .h-remove-elements': '_removeElements',
        'click .h-edit-elements': '_editElements',
        'click .h-set-group': '_setGroup',
        'click .h-remove-group': '_removeGroup'
    },
    initialize(settings) {
        this._cachedGroupCount = {};
        this.styles = new StyleCollection();
        this.styles.fetch().done(() => this.render());
        this.listenTo(this.collection, 'add remove reset', this.render);
    },
    render() {
        this.$el.html(template({
            groups: this._getAnnotationGroups(),
            numberSelected: this.collection.length
        }));
        return this;
    },
    refetchStyles() {
        this.styles.fetch().done(() => this.render());
    },
    setGroupCount(groupCount) {
        this._cachedGroupCount = groupCount;
    },
    _removeElements(evt) {
        evt.preventDefault();
        evt.stopPropagation();

        this.collection.trigger('h:remove');
        this.trigger('h:close');
    },
    _editElements(evt) {
        evt.preventDefault();
        evt.stopPropagation();

        this.trigger('h:edit', this.collection.at(0));
        this.trigger('h:close');
    },
    _setStyleDefinition(group) {
        const style = this.styles.get({ id: group }) || this.styles.get({ id: 'default' });
        const styleAttrs = Object.assign({}, style.toJSON());
        delete styleAttrs.id;
        this.collection.each((element) => { /* eslint-disable backbone/no-silent */
            if (group) {
                styleAttrs.group = group;
            } else {
                element.unset('group', {silent: true});
            }
            element.set(styleAttrs, {silent: true});
        });
        this.collection.trigger('h:save');
        this.trigger('h:close');
    },
    _getAnnotationGroups() {
        const groups = this.styles.map((style) => style.id);
        groups.sort((a, b) => {
            const countA = this._cachedGroupCount[a] || 0;
            const countB = this._cachedGroupCount[b] || 0;
            if (countA !== countB) {
                return countB - countA;
            }
            if (a < b) {
                return -1;
            } else if (a > b) {
                return 1;
            } else {
                return 0;
            }
        });
        return groups;
    },
    _setGroup(evt) {
        evt.preventDefault();
        evt.stopPropagation();

        const group = $(evt.currentTarget).data('group');
        this._setStyleDefinition(group);
    },
    _removeGroup(evt) {
        evt.preventDefault();
        evt.stopPropagation();

        this._setStyleDefinition(null);
    }
});

export default AnnotationContextMenu;
